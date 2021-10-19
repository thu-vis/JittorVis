import numpy as np

class JittorNetworkProcessor(object):

    def __init__(self):
        super().__init__()
        self.network = None
    
    def process(self, rawData: dict) -> dict:
        """main process function"""
        self.network = rawData
        self.network = self.mergeVarNode(self.network)
        self.network = self.constructHierarchy(self.network)
        self.network["branch"] = self.connectBranchNodes(self.network["branch"], self.network["leaf"])
        self.network["branch"] = self.setAttrsForBranchNodes(self.network["branch"], self.network["leaf"])
        self.network["branch"] = self.topoSortBranchNodes(self.network["branch"], self.network["leaf"])
        return self.network

    def mergeVarNode(self, network: dict) -> dict:
        """Merge variable node into operation node

        Args:
            network (dict): {nodeID: nodeInfo}

        Returns:
            dict: {nodeID: nodeInfo}
        """               
        newNetwork = {}
        for nodeID, node in network.items():
            if node["attrs"]["is_var"] == '0': 
                # assert outputs/inputs nodes are var nodes
                nextnodesID = node["outputs"]
                for varnodeID in nextnodesID:
                    assert network[varnodeID]["attrs"]["is_var"]
                prevnodesID = node["inputs"]
                for varnodeID in prevnodesID:
                    assert network[varnodeID]["attrs"]["is_var"]

                # new outputs is the concat of outputs of all next varnodes
                outputs = sum([network[varnodeID]["outputs"] for varnodeID in nextnodesID], [])
                # new inputs is the concat of outputs of all prev varnodes
                inputs = sum([network[varnodeID]["inputs"] for varnodeID in prevnodesID], [])


                # new attrs is the merge of all varnodes and opnode
                attrs = {}
                for varnodeID in nextnodesID:
                    attrs.update(network[varnodeID]["attrs"])
                attrs.update(node["attrs"])

                # new inputs is the concat of inputs of all last nodes


                newNetwork[nodeID] = {
                    "inputs": inputs,
                    "outputs": outputs,
                    "stacks": node["stacks"],
                    "attrs": attrs
                }

        return newNetwork

    def constructHierarchy(self, network: dict) -> dict:
        """Construct hierarchy from leaf nodes

        Args:
            network (dict): {nodeID: nodeInfo}

        Returns:
            dict: {leaf: {nodeID: nodeInfo}, branch: {nodeID: nodeInfo}}
        """
        # remove stack
        newNetwork = {}
        for nodeID, node in network.items():
            newNetwork[nodeID] = {
                "id": nodeID,
                "inputs": node["inputs"],
                "outputs": node["outputs"],
                "attrs": node["attrs"]
            }

        # get branch nodes
        branchNetwork = {}
        for nodeID, node in network.items():
            branchNodeKey = ""
            stacks = []
            if len(node["stacks"])==0:
                continue
            for i in range(len(node["stacks"])):
                branchNode = node["stacks"][i]
                branchNodeKey += branchNode["name"]+"/"
                stacks.append(branchNodeKey)
                if branchNodeKey not in branchNetwork:
                    branchNetwork[branchNodeKey] = {
                        "id": branchNodeKey,
                        "attrs": branchNode,
                        "children": []
                    }
            stacks.append(nodeID)
            for i in range(1, len(stacks)):
                if stacks[i] not in branchNetwork[stacks[i-1]]["children"]:
                    branchNetwork[stacks[i-1]]["children"].append(stacks[i])
                    if type(stacks[i])==int:
                        newNetwork[stacks[i]]["parent"] = stacks[i-1]
                    else:
                        branchNetwork[stacks[i]]["parent"] = stacks[i-1]
        
        # process when children of a branch node both have a branch node and a leaf node
        newBranchNodes = {}
        for branchID, branchNode in branchNetwork.items():
            assert len(branchNode["children"])>0
            isBottom = type(branchNode["children"][0])==int
            needProcess = False
            for childNodeID in branchNode["children"]:
                if (isBottom and type(childNodeID)!=int) or ((not isBottom) and type(childNodeID)==int):
                    needProcess = True
                    break
            if needProcess:
                for i in range(len(branchNode["children"])):
                    if type(branchNode["children"][i])==int:
                        childNode = newNetwork[branchNode["children"][i]]
                        newBranchNode = {
                            "id": branchNode["id"]+childNode["attrs"]["name"]+'/',
                            "children": [branchNode["children"][i]],
                            "attrs": {
                                "name": childNode["attrs"]["name"],
                                "type": childNode["attrs"]["dtype"],
                            },
                            "parent": branchID
                        }
                        childNode["parent"] = newBranchNode["id"]
                        branchNode["children"][i] = newBranchNode["id"]
                        newBranchNodes[newBranchNode["id"]] = newBranchNode

        return {
            "leaf": newNetwork,
            "branch": {**branchNetwork, **newBranchNodes}
        }

    def connectBranchNodes(self, branch: dict, leaf: dict) -> dict:
        """connect branch nodes

        Args:
            branch (dict): branchnodes
            leaf (dict): leafnodes

        Returns:
            dict: new branch
        """               
        for branchID, branchNode in branch.items():
            assert len(branchNode["children"])>0
            if type(branchNode["children"][0])==int:
                branchNode["inputs"] = set()
                branchNode["outputs"] = set()

        for branchID, branchNode in branch.items():
            if type(branchNode["children"][0])!=int:
                continue
            for child in branchNode["children"]:
                for inputchild in leaf[child]["inputs"]:
                    if ("parent" not in leaf[inputchild]) or (leaf[inputchild]["parent"]==branchID):
                        continue
                    inputNode = leaf[inputchild]["parent"]
                    branchNode["inputs"].add(branch[inputNode]["id"])
                    branch[inputNode]["outputs"].add(branchID)
        
        for branchID, branchNode in branch.items():
            assert len(branchNode["children"])>0
            if type(branchNode["children"][0])==int:
                branchNode["inputs"] = list(branchNode["inputs"])
                branchNode["outputs"] = list(branchNode["outputs"])

        return branch

    def setAttrsForBranchNodes(self, branch: dict, leaf: dict) -> dict:
        """As branch nodes are created with lots of attrs hidden in leaf nodes, set attributes for branch nodes

        Args:
            branch (dict): branch nodes
            leaf (dict): leaf nodes

        Returns:
            dict: new branch nodes
        """   
        for branchNodeID, branchNode in branch.items():
            branchNode["name"] = branchNode["attrs"]["name"]
            branchNode["type"] = branchNode["attrs"]["type"]
            newAttrs = {}
            # add conv kernal size and channel num for conv layer     
            if branchNode["attrs"]["type"] == "Conv":
                assert type(branchNode["children"][0]) == int
                for childID in branchNode["children"]:
                    if leaf[childID]["attrs"]["name"] == "binary.multiply":
                        shape = [num for num in leaf[childID]["attrs"]["shape"][1:len(leaf[childID]["attrs"]["shape"])-2].split(',')]
                        newAttrs["input"] = '×'.join(shape[2:5])
                        newAttrs["kernal"] = '×'.join(shape[5:7])
                        newAttrs["channels"] = shape[1]

            # add name for sequential layer
            if branchNode["attrs"]["type"] == "Sequential":
                newAttrs["name"] = branchNode["attrs"]["name"]

            # add size for fc layer
            if branchNode["attrs"]["type"] == "Linear":
                assert type(branchNode["children"][0]) == int
                for childID in branchNode["children"]:
                    if leaf[childID]["attrs"]["name"] == "binary.multiply":
                        inputDim = int(leaf[childID]["attrs"]["shape"].split(',')[2])
                        newAttrs['inputDim'] = inputDim
                    if leaf[childID]["attrs"]["name"] == "binary.add":
                        outputDim = int(leaf[childID]["attrs"]["shape"].split(',')[1])
                        newAttrs['outputDim'] = outputDim

            if branchNode["attrs"]["type"].startswith('float'):
                branchNode["type"] = branchNode["name"]


            branchNode["attrs"] = newAttrs
        return branch

    def topoSortBranchNodes(self, branch: dict, leaf: dict) -> dict:
        """sort branch nodes' children

        Args:
            branch (dict): branch nodes
            leaf (dict): leaf nodes

        Returns:
            dict: new branch nodes
        """
        network = {**branch, **leaf}
        parent2children = {}
        def sortIter(node):
            if node in leaf:
                #if leaf node, return
                return
            for childid in network[node]['children']:
                sortIter(childid)

            inputs = {}
            children2parent = {}
            isbottom = len(network[node]['children'])==0 or (network[node]['children'][0] in leaf)

            # init children
            if isbottom:
                parent2children[node] = network[node]['children']
            else:
                parent2children[node] = []
                for childid in network[node]['children']:
                    parent2children[node] += parent2children[childid]

            if isbottom:
                for id in network[node]['children']:
                    inputs[id] = network[id]['inputs']
            else:
                for id in network[node]['children']:
                    for cid in parent2children[id]:
                        children2parent[cid] = id
                for id in network[node]['children']:
                    inputs[id] = []
                    for cid in parent2children[id]:
                        for lastid in network[cid]['inputs']:
                            if (lastid in children2parent) and (children2parent[lastid]!=id) and (children2parent[lastid] not in inputs[id]):
                                inputs[id].append(children2parent[lastid])
            topos = []
            childlen = len(network[node]['children'])
            for i in range(childlen):
                for id in network[node]['children']:
                    if id in topos:
                        continue
                    flag = True
                    for inputid in inputs[id]:
                        if (inputid in network[node]['children']) and (inputid not in topos):
                            flag = False
                            break
                    if flag:
                        topos.append(id)
            network[node]['children'] = topos

        # find root
        root = None
        for id, node in branch.items():
            if 'parent' not in node:
                root = id
                break
        sortIter(root)
        return branch

if __name__ == "__main__":
    import pickle
    jittorProcessor = JittorNetworkProcessor()
    with open("/home/zhaowei/JittorModels/Jittor-Image-Models/resnet26.pkl", "rb") as f:
        rawData = pickle.load(f)
    jittorProcessor.process(rawData["node_data"])
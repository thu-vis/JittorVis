class JittorNetworkProcessor(object):

    def __init__(self):
        super().__init__()
        self.network = None
    
    def process(self, rawData: dict) -> dict:
        """main process function"""
        self.network = rawData
        self.network = self.__mergeVarNode(self.network)
        self.network = self.__constructHierarchy(self.network)
        return self.network

    def __mergeVarNode(self, network: dict) -> dict:
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

    def __constructHierarchy(self, network: dict) -> dict:
        """Construct hierarchy from leaf nodes

        Args:
            network (dict): {nodeID: nodeInfo}

        Returns:
            dict: {nodeID: nodeInfo}
        """
        newNetwork = network.copy()
        for nodeID, node in network.items():
            branchNodeKey = ""
            stacks = []
            for i in range(len(node["stacks"])):
                branchNode = node["stacks"][i]
                branchNodeKey += branchNode["name"]+"/"
                stacks.append(branchNodeKey)
                if branchNodeKey not in newNetwork:
                    newNetwork[branchNodeKey] = {
                        "id": branchNodeKey,
                        "attrs": branchNode,
                        "children": []
                    }
            stacks.append(nodeID)
            for i in range(1, len(stacks)):
                newNetwork[stacks[i-1]]["children"].append(stacks[i])
                newNetwork[stacks[i]]["parent"] = stacks[i-1]
        
        return newNetwork


if __name__ == "__main__":
    import pickle
    jittorProcessor = JittorNetworkProcessor()
    with open("/home/zhaowei/JittorModels/Jittor-Image-Models/resnet18.pkl", "rb") as f:
        rawData = pickle.load(f)
    jittorProcessor.process(rawData["node_data"])
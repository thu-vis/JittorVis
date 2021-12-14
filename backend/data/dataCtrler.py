from data.network import JittorNetworkProcessor
import copy
import os
import numpy as np
from PIL import Image
import math
from tempfile import NamedTemporaryFile
from data.sampling import HierarchySampling
from data.gridLayout import GridLayout

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.networkRawdata = {}
        self.network = {}
        self.statistic = {}
        self.labels = None
        self.preds = None
        self.features = None
        
        self.grider = GridLayout()
        self.sampler = HierarchySampling()

    def processNetworkData(self, network: dict) -> dict:
        processor = JittorNetworkProcessor()
        return processor.process(network)

    def processStatisticData(self, data):     
        return data
    
    def processSamplingData(self, samplingPath):
        self.sampling_buffer_path = samplingPath
        
        if os.path.exists(self.sampling_buffer_path):
            self.sampler.load(self.sampling_buffer_path)
        else:
            data = self.features
            n = data.shape[0]
            d = 1
            for dx in data.shape[1:]:
                d *= dx
            data = data.reshape((n, d))
            
            labels = self.labels
            self.sampler.fit(data, labels, 0.25, 1600)
            self.sampler.dump(self.sampling_buffer_path)

    def process(self, networkRawdata, statisticData, predictData = None, modeltype='jittor', trainImages = None, sampling_buffer_path="/tmp/hierarchy.pkl", attrs = {}):
        """process raw data
        """        
        self.networkRawdata = networkRawdata
        self.network = self.processNetworkData(self.networkRawdata["node_data"])
        self.statistic = self.processStatisticData(statisticData)

        if predictData is not None:
            self.labels = predictData["labels"].astype(int)
            self.preds = predictData["preds"].astype(int)
            self.features = predictData["features"]
            self.sampling = self.processSamplingData(sampling_buffer_path)
        self.trainImages = trainImages
        
    def getBranchTree(self) -> dict:
        """get tree of network
        """        
        branch = self.network["branch"]
        newBranch = copy.deepcopy(branch)
        for branchID, branchNode in newBranch.items():
            if type(branchNode["children"][0])==int:
                branchNode["children"]=[]
        return newBranch

    def getBranchNodeOutput(self, branchID: str) -> np.ndarray:
        """unserializae leaf data from str to numpy.ndarray

        Args:
            branchID (str): branch node id

        Returns:
            np.ndarray: branch node output
        """        
        # first, get output opr node
        def visitTree(root):
            outputnode = None
            for childID in root["children"]:
                if type(childID)==int:
                    if "data" in self.network["leaf"][childID]["attrs"]:
                        outputnode = self.network["leaf"][childID]
                else:
                    childOutput = visitTree(self.network["branch"][childID])
                    if childOutput:
                        outputnode = childOutput
            return outputnode
        outputnode = visitTree(self.network["branch"][branchID])
        if outputnode is None:
            return {
                "leafID": -1
            }
        # second, compute the data
        if type(outputnode["attrs"]["data"]) == str:
            shape = [int(num) for num in outputnode['attrs']['shape'][1:len(outputnode['attrs']['shape'])-2].split(',')]
            data = np.array([float(num) for num in outputnode['attrs']['data'].split(',')])
            data = data.reshape(tuple(shape))
            if int(outputnode["attrs"]["ndim"])>1:
                data = data[0]
                shape = shape[1:]
            outputnode["attrs"]["data"] = data
            outputnode["attrs"]["shape"] = shape

        # third, get all 
        if len(outputnode["attrs"]["shape"])==1:
            features = [self.getFeature(outputnode["id"], -1)]
            maxActivations = []
            minActivations = []
        else:
            features = [self.getFeature(outputnode["id"], featureIndex) for featureIndex in range(outputnode["attrs"]["shape"][0])]
            maxActivations = [float(np.max(feature)) for feature in features]
            minActivations = [float(np.min(feature)) for feature in features]

        return {
            "leafID": outputnode["id"],
            "shape": outputnode["attrs"]["shape"],
            "features": features,
            "maxActivations": maxActivations,
            "minActivations": minActivations
        }

    def getFeature(self, leafID, featureIndex: int) -> str:
        """get feature map of a opr node

        Args:
            leafID (int or str): opr node id
            featureIndex (int): feature map index, if -1, return whole feature

        Returns:
            list: feature map image path
        """
        # compute the data
        leafNode = self.network["leaf"][leafID]
        if type(leafNode["attrs"]["data"]) == str:
            shape = [int(num) for num in leafNode['attrs']['shape'][1:len(leafNode['attrs']['shape'])-2].split(',')]
            data = np.array([float(num) for num in leafNode['attrs']['data'].split(',')])
            data = data.reshape(tuple(shape))
            if int(leafNode["attrs"]["ndim"])>1:
                data = data[0]
                shape = shape[1:]
            leafNode["attrs"]["data"] = data
            leafNode["attrs"]["shape"] = shape

        # get feature
        feature = None
        if featureIndex==-1:
            feature = self.network["leaf"][leafID]["attrs"]["data"]
        else:
            feature = self.network["leaf"][leafID]["attrs"]["data"][featureIndex]
        if len(feature.shape)==1:
            width = math.ceil(np.sqrt(feature.shape[0]))
            if feature.shape[0] < width*width:
                zeroPad = np.zeros((width*width-feature.shape[0]))
                feature = np.concatenate((feature, zeroPad))
            feature = feature.reshape((width, width))
        
        return feature.tolist()

    def getStatisticData(self):
        """get statistic data
        """
        return self.statistic        

    def getConfusionMatrix(self):
        """ confusion matrix
        """        
        return self.statistic["confusion"]
    
    def getImagesInConsuionMatrixCell(self, labels: list, preds: list) -> list:
        """return images in a cell of confusionmatrix

        Args:
            labels (list): true labels of corresponding cell
            preds (list): predicted labels of corresponding cell

        Returns:
            list: images' id
        """ 
        # convert list of label names to dict
        labelNames = self.statistic['confusion']['names']
        name2idx = {}
        for i in range(len(labelNames)):
            name2idx[labelNames[i]]=i
        
        # find images
        labelSet = set()
        for label in labels:
            labelSet.add(name2idx[label])
        predSet = set()
        for label in preds:
            predSet.add(name2idx[label])
        imageids = []
        if self.labels is not None and self.preds is not None:
            n = len(self.labels)
            for i in range(n):
                if self.labels[i] in labelSet and self.preds[i] in predSet:
                    imageids.append(i)
                    
        # limit length of images
        return imageids[:50]
    
    def getImageGradient(self, imageID: int) -> list:
        """ get gradient of image

        Args:
            imageID (int): image id

        Returns:
            list: gradient
        """        
        if self.trainImages is not None:
            return self.trainImages[imageID].tolist()
        else:
            return []
        
    def gridZoomIn(self, nodes, constraints, depth):
        neighbors, newDepth = self.sampler.zoomin(nodes, depth)
        zoomInConstraints = None
        if constraints is not None:
            zoomInConstraints = []
        zoomInNodes = []
        if type(neighbors)==list:
            zoomInNodes = neighbors
        else:
            for i in range(len(nodes)):
                parent = nodes[i]
                for child in neighbors[parent]:
                    zoomInNodes.append(child)
                    if constraints is not None:
                        zoomInConstraints.append(constraints[i])
        zoomInLabels = self.labels[zoomInNodes]
        
        tsne, grid, gridsize = self.grider.fit(self.features[zoomInNodes], labels = zoomInLabels, constraintX = zoomInConstraints)
        tsne = tsne.tolist()
        grid = grid.tolist()
        zoomInLabels = zoomInLabels.tolist()
        n = len(zoomInNodes)
        nodes = [{
            "index": zoomInNodes[i],
            "tsne": tsne[i],
            "grid": grid[i],
            "label": zoomInLabels[i]
        } for i in range(n)]
        return {
            "nodes": nodes,
            "grid": {
                "width": gridsize,
                "height": gridsize,
            },
            "depth": newDepth
        }
dataCtrler = DataCtrler()
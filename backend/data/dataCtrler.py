from data.network import JittorNetworkProcessor
import copy
import os
import numpy as np
from PIL import Image
import math
from tempfile import NamedTemporaryFile

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.rawdata = {}
        self.network = {}
        self.statistic = {}

    def processNetworkData(self, network: dict) -> dict:
        processor = JittorNetworkProcessor()
        return processor.process(network)

    def processStatisticData(self, data):
        return {}

    def process(self, rawdata, modeltype='jittor', attrs = {}):
        """process raw data
        """        
        self.rawdata = rawdata
        self.network = self.processNetworkData(self.rawdata["node_data"])
        self.statistic = self.processStatisticData(self.rawdata)

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
        

        return {
            "leafID": outputnode["id"],
            "shape": outputnode["attrs"]["shape"]
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
        
        # feature process from https://www.analyticsvidhya.com/blog/2020/11/tutorial-how-to-visualize-feature-maps-directly-from-cnn-layers/
        feature-= feature.mean()
        feature/= feature.std ()
        feature*=  64
        feature+= 128
        feature= np.clip(feature, 0, 255).astype('uint8')

        # return image
        img = Image.fromarray(feature).convert("RGB")
        tmpfile = NamedTemporaryFile(suffix=".png", delete=False)
        imagePath = tmpfile.name
        img.save(imagePath)
        return imagePath

    def getStatisticData(self):
        """get statistic data
        """
        return self.statistic        

dataCtrler = DataCtrler()
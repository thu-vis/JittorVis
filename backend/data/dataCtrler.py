from data.network import JittorNetworkProcessor

class DataCtrler(object):

    def __init__(self):
        super().__init__()
        self.rawdata = {}
        self.__network = {}
        self.__statistic = {}

    def processNetworkData(self, network: dict) -> dict:
        processor = JittorNetworkProcessor()
        return processor.process(network)

    def processStatisticData(self, data):
        return {}

    def process(self, rawdata):
        """process raw data
        """        
        self.rawdata = rawdata
        self.__network = self.processNetworkData(self.rawdata["node_data"])
        self.__statistic = self.processStatisticData(self.rawdata)

    def getBranchTree(self) -> dict:
        """get tree of network
        """        
        branch = self.__network["branch"]
        return branch

    def getStatisticData(self):
        """get statistic data
        """
        return self.__statistic        

dataCtrler = DataCtrler()
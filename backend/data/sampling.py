import numpy as np
import pickle
import random
import faiss
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

# for efficient add, remove, and random select
# modified from https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
class ListDict(object):
    def __init__(self, items = []):
        self.items = items
        self.item_to_position = {}
        for i in range(len(items)):
            self.item_to_position[items[i]] = i
            
    def __len__(self):
        return len(self.items)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

class OutlierBiasedBlueNoiseSamplingFAISS():
    def __init__(self, sampling_rate, outlier_score=None, fail_rate=0.1):
        self.sampling_rate = sampling_rate
        self.outlier_score = outlier_score
        self.fail_rate = fail_rate

    def fit(self, data, category):
        data = np.array(data.tolist(), dtype=np.float32)
        n, d = data.shape
        
        allIndexer = faiss.IndexFlatL2(d)
        allIndexer.add(data)
        if self.outlier_score is None:
            self.outlier_score = get_default_outlier_scores(data, category, dataIndexer=allIndexer)
        prob = self.outlier_score / (2 * np.max(self.outlier_score)) + 0.5
        if type(self.sampling_rate)==float:
            m = round(n * self.sampling_rate)
        else:
            m = self.sampling_rate
            self.sampling_rate = m/n
        k = int(1 / self.sampling_rate)
        dist, _ = allIndexer.search(data, k+1)
        radius = np.average(np.sqrt(dist[:, -1]))

        selected_indexes = []

        count = 0
        candidates = ListDict(items=list(range(n)))
        indexer = faiss.IndexFlatL2(d)
        while count < m:
            failure_tolerance = min(1000, (n - m) * self.fail_rate)
            fail = 0
            for i in range(len(candidates)):
                idx = candidates.choose_random_item()
                if fail > failure_tolerance or count >= m:
                    break
                if random.random() < prob[idx]:
                    fail += 1
                    continue
                success = True
                topK = 1
                nearestDis, nearestIdx = indexer.search(data[idx:idx+1, :], 1)
                if nearestIdx[0][0]!=-1 and nearestDis[0][0]<radius:
                    success = False
                if success:
                    count += 1
                    selected_indexes.append(idx)
                    indexer.add(data[idx:idx+1,:])
                    candidates.remove_item(idx)
                else:
                    fail += 1
            radius /= 2

        selected_indexes = np.array(selected_indexes)
        _, nearestIdx = indexer.search(data, 1)
        nearestIdx = np.array(selected_indexes)[nearestIdx.reshape((n))]
        
        return selected_indexes, nearestIdx

def Knn(X, N, D, n_neighbors, forest_size, subdivide_variance_size, leaf_number):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, leaf_size=leaf_number)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    return indices, distances

def get_default_outlier_scores(data, category, k=50, dataIndexer = None):
    X = np.array(data.tolist(), dtype=np.float64)
    n, d = X.shape
    if k + 1 > n:
        k = int((n - 1) / 2)
    if dataIndexer is not None:
        distances, neighbor = dataIndexer.search(data, k+1)
    else:
        neighbor, _ = Knn(X, n, d, k + 1, 1, 1, n)
    neighbor_labels = category[neighbor]
    outlier_score = [sum(neighbor_labels[i] != category[i]) for i in range(data.shape[0])]
    outlier_score = np.array(outlier_score) / k
    return outlier_score

class HierarchySampling(object):
    
    def __init__(self):
        super().__init__()
        self.hierarchy = {}
        self.max_depth = 0
        self.top_nodes = []
        
    def fit(self, data, category, sampling_rate, top_nodes_count):
        n = data.shape[0]
        self.hierarchy = [[[]] for i in range(n)]
        self.max_depth = 1
            
        selected_indexes = np.arange(n)
        while len(selected_indexes)>top_nodes_count:
            sampling_number = int(max(top_nodes_count, len(selected_indexes)*sampling_rate))
            logging.info("hierarchy sampling from {} to {}, depth = {}".format(len(selected_indexes), sampling_number, self.max_depth))
            faiss_sampler = OutlierBiasedBlueNoiseSamplingFAISS(sampling_rate=sampling_number)
            new_selected_indexes, neighbors = faiss_sampler.fit(data[selected_indexes], category[selected_indexes])
            new_selected_indexes = selected_indexes[new_selected_indexes].tolist()
            neighbors = selected_indexes[neighbors].tolist()
            for new_selected_index in new_selected_indexes:
                self.hierarchy[new_selected_index].append([])
            self.max_depth += 1
            for i in range(len(selected_indexes)):
                self.hierarchy[neighbors[i]][self.max_depth-1].append(selected_indexes[i])
            selected_indexes = np.array(new_selected_indexes)
            
        self.top_nodes = selected_indexes.tolist()
        
    def zoomin(self, indexes, depth):
        if depth==0:
            return self.top_nodes, 1
        else:
            selected_indexes = {}
            depth += 1
            assert depth<=self.max_depth
            for index in indexes:
                selected_indexes[index] = self.hierarchy[index][self.max_depth-depth]
            return selected_indexes, depth
        
    def dump(self, path):
        with open(path, "wb") as file:
            pickle.dump({
                "hierarchy": self.hierarchy,
                "max_depth": self.max_depth,
                "top_nodes": self.top_nodes
                }, file)
            
    def load(self, path):
        with open(path, "rb") as file:
            hierarchyInfo = pickle.load(file)
            self.hierarchy = hierarchyInfo["hierarchy"]
            self.max_depth = hierarchyInfo["max_depth"]
            self.top_nodes = hierarchyInfo["top_nodes"]
            
import numpy as np
import random
import faiss
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

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

class OutlierBiasedBlueNoiseSampling():
    def __init__(self, sampling_rate, outlier_score=None, fail_rate=0.1):
        self.sampling_rate = sampling_rate
        self.outlier_score = outlier_score
        self.fail_rate = fail_rate

    def fit(self, data, category):
        if self.outlier_score is None:
            self.outlier_score = get_default_outlier_scores(data, category)
        prob = self.outlier_score / (2 * np.max(self.outlier_score)) + 0.5

        X = np.array(data.tolist(), dtype=np.float64)
        n, d = X.shape
        m = round(n * self.sampling_rate)
        k = int(1 / self.sampling_rate)
        neighbor, dist = Knn(X, n, d, k + 1, 1, 1, n)
        radius = np.average(np.sqrt(dist[:, -1]))

        selected_indexes = []

        count = 0
        candidates = ListDict(items=list(range(n)))
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
                for selected_id in selected_indexes:
                    if sum((data[idx] - data[selected_id])**2) < radius**2:
                        success = False
                        break
                if success:
                    count += 1
                    selected_indexes.append(idx)
                    candidates.remove_item(idx)
                else:
                    fail += 1
            radius /= 2

        selected_indexes = np.array(selected_indexes)
        return selected_indexes

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
        return selected_indexes


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

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    import os
    import pickle
    import time
    
    # def tsne(X: np.ndarray, perplexity: int = 15, learning_rate: int = 3) -> np.ndarray:
    #     X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(X)
    #     return X_embedded
    
    # def _draw_tsne(X, labels, title):
    #     label_colors = ["#A9A9A9", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    #         "#ffdb45", "#bcbd22", "#17becf"]
    #     colors = [label_colors[label] for label in labels]
    #     plt.figure()
    #     plt.scatter(X[:, 0], X[:, 1], s=2, c=colors)
    #     plt.savefig('{}-tsne.png'.format(title))
    
    # random.seed(0)
    # with open('/data/zhaowei/jittor-data/predict_info.pkl', 'rb') as f:
    #     predictData = pickle.load(f)
    # features = predictData['features']
    # labels = predictData['labels']
    # embedded_path = '/data/zhaowei/jittor-data/buffer/featuresEmbeded.npy'
    # if os.path.exists(embedded_path):
    #     embedded = np.load(embedded_path)
    # else:
    #     embedded = tsne(features)
    #     np.save(embedded_path, embedded)
    # indexes = np.load('/data/zhaowei/jittor-data/buffer/sample.npy')
    # _draw_tsne(embedded, labels, 'all')
    n = 100000
    features = np.random.rand(n, 2048)
    labels = np.random.randint(10, size=n)
    t0 = time.time()
    sampler = OutlierBiasedBlueNoiseSamplingFAISS(0.1)
    selectedIdxes = sampler.fit(features, labels)
    print("sampling time", time.time()-t0)
    np.save('/data/zhaowei/jittor-data/buffer/faiss_sample.npy', selectedIdxes)
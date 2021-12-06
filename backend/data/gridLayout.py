import numpy as np
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
import fastlapjv
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from fastlapjv import fastlapjv
import math
from time import time

from data.sampling import OutlierBiasedBlueNoiseSamplingFAISS

class GridLayout(object):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, labels: np.ndarray):
        """main fit function

        Args:
            X (np.ndarray): n * d, n is the number of samples, d is the dimension of a sample
            labels (np.ndarray): label of each sample in X
        """        
        n = X.shape[0]
        if n>1600:
            sampledIndeces = self.sample(X, labels, 1600)
            X = X[sampledIndeces]
            labels = labels[sampledIndeces]
        X_embedded = self.tsne(X)
        # self._draw_tsne(X_embedded, labels)
        grid_ass, grid_size = self.grid(X_embedded)
        
    def sample(self, X: np.ndarray, labels: np.ndarray, sampleNumber: int) -> np.ndarray:
        """sample from X

        Args:
            X (np.ndarray): n * d, n is the number of samples, d is the dimension of a sample
            labels (np.ndarray): label of each sample in X
            sampleNumber (int): number to be sampled

        Returns:
            np.ndarray: indices of sampled result
        """
        sampler = OutlierBiasedBlueNoiseSamplingFAISS(1600)
        indices = sampler.fit(X, labels)
        return indices
    
    def tsne(self, X: np.ndarray, perplexity: int = 15, learning_rate: int = 3) -> np.ndarray:
        X_embedded = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate).fit_transform(X)
        return X_embedded
    
    def grid(self, X_embedded: np.ndarray):
        X_embedded -= X_embedded.min(axis=0)
        X_embedded /= X_embedded.max(axis=0)
        num = X_embedded.shape[0]
        square_len = math.ceil(np.sqrt(num))
        N = square_len * square_len
        grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                np.linspace(0, 1 - 1.0 / square_len, square_len))) \
                .reshape(-1, 2)

        original_cost_matrix = cdist(grids, X_embedded, "euclidean")
        # knn process
        dummy_points = np.ones((N - original_cost_matrix.shape[1], 2)) * 0.5
        # dummy at [0.5, 0.5]
        dummy_vertices = (1 - cdist(grids, dummy_points, "euclidean")) * 100
        cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)
        row_asses, col_asses, info = fastlapjv(cost_matrix, k_value=50)
        col_asses = col_asses[:num]
        return col_asses, square_len
    
    def _draw_tsne(self, X, labels):
        labels = labels.astype(int)
        label_colors = ["#A9A9A9", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
            "#ffdb45", "#bcbd22", "#17becf"]
        colors = [label_colors[label] for label in labels]
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=2, c=colors)
        plt.savefig('tsne.png')
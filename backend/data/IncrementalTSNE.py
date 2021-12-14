from __future__ import print_function
from glob import glob
import threading
import os
import platform
import sys
import scipy.sparse as sp
from sklearn.utils import check_array, check_random_state
from sklearn.decomposition import PCA
import numpy as np
import cffi
from time import time
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from scipy.spatial.distance import sqeuclidean
from scipy.spatial.distance import squareform
from sklearn.manifold import _barnes_hut_tsne
from sklearn.metrics.pairwise import pairwise_distances
from annoy import AnnoyIndex
import math

MACHINE_EPSILON = np.finfo(np.double).eps

Global = {}


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        return self._target(*self._args)


class IncrementalTSNE:
    """
    Incremental t-distributed Stochastic Neighbor Embedding.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Dimension of the embedded space.

    perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.

    early_exaggeration : float, optional (default: 12.0)
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float, optional (default: 200.0)
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.

    n_iter : int, optional (default: 1000)
        Maximum number of iterations for the optimization. Should be at
        least 250.

    n_iter_without_progress : int, optional (default: 300)
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, optional (default: 1e-7)
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : string or callable, optional
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    init : string or numpy array, optional (default: "random")
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

    verbose : int, optional (default: 0)
        Verbosity level.

    random_state : int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.  Note that different initializations might result in
        different local minima of the cost function.

    method : string (default: 'barnes_hut')
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float (default: 0.5)
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    Attributes
    ----------
    embedding_ : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------

    # >>> import numpy as np
    # >>> from sklearn.manifold import TSNE
    # >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # >>> X_embedded = TSNE(n_components=2).fit_transform(X)
    # >>> X_embedded.shape
    (4, 2)
    """
    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0,
                 learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
                 min_grad_norm=1e-7, metric='euclidean', init='random', verbose=0,
                 random_state=None, method='barnes_hut', angle=0.5, n_jobs=1, accuracy=1.0, exploration_n_iter=250):
        self.n_components = n_components
        self._EXPLORATION_N_ITER = exploration_n_iter
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.accuracy = accuracy
        assert 0.01 < accuracy <= 1.0, "accuracy should in (0.01, 1]"
        assert isinstance(init, np.ndarray) or init == 'random' or init == 'pca', "init must be random, pca or array"

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X, double* constraint_Y,
			                    int constraint_N, double alpha, double perplexity, double angle, int n_jobs,
                                int n_iter, int random_state, int verbose, double accuracy, double early_exaggeration, double learning_rate,
                                int skip_num_points, int exploration_n_iter,
                                int n_neighbors, int* neighbors_nn, int* constraint_neighbors_nn, double* distances_nn, double* constraint_distances_nn, double* constraint_weight, int last_n);
              void binary_search_perplexity(double* distances_nn, int* neighbors_nn, int N, int D,
                                double perplexity, int verbose, double* conditional_P);
              void k_neighbors(double* X1, int N1, double* X2, int N2, int D, int n_neighbors, int* neighbors_nn, double* distances_nn,
                                int forest_size, int subdivide_variance_size, int leaf_number, int knn_tree,
                                int verbose);
              void multi_run_bhtsne(double* X, int N, int D, double* Y, int no_dim, double* constraint_X,
                                double* constraint_Y, int constraint_N, double alpha, double perplexity,
                                double angle, int n_jobs, int n_iter, int random_state, int verbose,
                                double accuracy, double early_exaggeration, double learning_rate,
                                int skip_num_points, int exploration_n_iter);
             """)

        path = os.path.dirname(os.path.realpath(__file__))


        try:
            if platform.system() == "Windows":
                sofile = glob(os.path.join(path, 'libtsne_incremental.dll'))[0]
            else:
                sofile = glob(os.path.join(path, 'libtsne_incremental.so'))[0]
                #raise RuntimeError('No so build now')
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open tsne_incremental shared library')

    def fit(self, X, y=None):
        """
        Fit X into an embedded space.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None, skip_num_points=0, constraint_X=None, constraint_Y=None, alpha=0,
                      corrected_id=None, corrected_weight=None, init_id=None, labels=None, prev_n=0,
                      constraint_labels=None, label_alpha=0.6, constraint_weight=None):
        """
        Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        if constraint_X is not None:
            constraint_X = constraint_X.astype(np.float64, copy=True)
        if constraint_Y is not None:
            constraint_Y = constraint_Y.astype(np.float64, copy=True)
        if corrected_id is not None:
            corrected_id = corrected_id.astype(np.int32, copy=True)
        if corrected_weight is not None:
            corrected_weight = corrected_weight.astype(np.float64, copy=True)
        if init_id is not None:
            init_id = init_id.astype(np.int32, copy=True)

        embedding = self._fit(X, skip_num_points=skip_num_points, constraint_X=constraint_X, constraint_Y=constraint_Y,
                              alpha=alpha, corrected_id=corrected_id, corrected_weight=corrected_weight, init_id=init_id, labels=labels, prev_n=prev_n,
                              constraint_labels=constraint_labels, label_alpha=label_alpha, constraint_weight=constraint_weight)
        self.embedding_ = embedding
        return self.embedding_

    def _fit(self, X, skip_num_points=0, constraint_X=None, constraint_Y=None, alpha=0, corrected_id=None, corrected_weight=None, init_id=None, constraint_labels=None, labels=None, label_alpha=1, constraint_weight=None, prev_n=0):
        """
        Fit the model using X as training data.

        Note that sparse arrays can only be handled by method='exact'.
        It is recommended that you convert your sparse array to dense
        (e.g. `X.toarray()`) if it fits in memory, or otherwise using a
        dimensionality reduction technique (e.g. TruncatedSVD).

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. Note that this
            when method='barnes_hut', X cannot be a sparse array and if need be
            will be converted to a 32 bit float array. Method='exact' allows
            sparse arrays and 64bit floating point inputs.

        skip_num_points : int (optional, default:0)
            This does not compute the gradient for points with indices below
            `skip_num_points`. This is useful when computing transforms of new
            data where you'd like to keep the old data fixed.
        """
        if X.shape[1] == self.n_components:
            return X
        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            if np.any(X < 0):
                raise ValueError("All distances should be positive, the "
                                 "precomputed distances given as X is not "
                                 "correct")
        if self.method == 'barnes_hut' and sp.issparse(X):
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required for method="barnes_hut". Use '
                            'X.toarray() to convert to a dense numpy array if '
                            'the array is small enough for it to fit in '
                            'memory. Otherwise consider dimensionality '
                            'reduction techniques (e.g. TruncatedSVD)')
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=[np.float32, np.float64])
        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < self._EXPLORATION_N_ITER:
            raise ValueError("n_iter should be at least {}".format(self._EXPLORATION_N_ITER))

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric)

                if np.any(distances < 0):
                    raise ValueError("All distances should be positive, the "
                                     "metric given is not correct")

            # compute the joint probability distribution for the input space
            P = self.joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                    "or then equal to one")
            if isinstance(self.init, np.ndarray):
                X_embedded = self.init.astype(np.float64, copy=True)
            elif self.init == 'pca':
                pca = PCA(n_components=self.n_components, svd_solver='randomized',
                          random_state=random_state)
                X_embedded = pca.fit_transform(X).astype(np.float64, copy=False)
            elif self.init == 'random':
                # The embedding is initialized with iid samples from Gaussians with
                # standard deviation 1e-4.
                X_embedded = 1e-4 * random_state.randn(
                    n_samples, self.n_components).astype(np.float64)
            else:
                raise ValueError("'init' must be 'pca', 'random', or "
                                 "a numpy array")

            # Degrees of freedom of the Student's t-distribution. The suggestion
            # degrees_of_freedom = n_components - 1 comes from
            # "Learning a Parametric Embedding by Preserving Local Structure"
            # Laurens van der Maaten, 2009.
            degrees_of_freedom = max(self.n_components - 1.0, 1)

            return self._tsne(P, degrees_of_freedom, n_samples, random_state,
                              X_embedded=X_embedded,
                              neighbors=neighbors_nn,
                              skip_num_points=skip_num_points)

        else:
            N, D = X.shape
            if labels is None:
                labels = np.ones(N) * -1
            n_labels = int(labels.max() + 1)
            means = []
            means_embed = []
            distribution = {}
            if n_labels > 0:
                for i in range(n_labels):
                    distribution[i] = [j for j in range(N) if labels[j] == i]
                    means.append(X[distribution[i]].mean(axis = 0))

            if isinstance(self.init, np.ndarray):
                Y = self.init.astype(np.float64, copy=True)
            elif self.init == 'pca':
                pca = PCA(n_components=self.n_components, svd_solver='randomized',
                          random_state=random_state)
                Y = pca.fit_transform(X).astype(np.float64, copy=False)
            elif self.init == 'random':
                # The embedding is initialized with iid samples from Gaussians with
                # standard deviation 1e-4.
                Y = 1e-4 * random_state.randn(
                    n_samples, self.n_components).astype(np.float64)
            else:
                raise ValueError("'init' must be 'pca', 'random', or "
                                 "a numpy array")

            if n_labels > 0:
                for i in range(n_labels):
                    means_embed.append(Y[distribution[i]].mean(axis = 0))

            cffi_verbose = 0
            if self.verbose:
                cffi_verbose = 1
            if labels is None:
                labels = np.zeros((N, ), dtype=int)
                label_alpha = 1.0

            if n_labels > 0:
                labels = np.concatenate((labels, [i for i in range(n_labels)]))
                X = np.concatenate((np.array(X, copy=True), np.array(means, copy=True)))
                Y = np.concatenate((np.array(Y, copy=True), np.array(means_embed, copy=True)))
            forest = AnnoyIndex(X.shape[1], 'euclidean')
            indices = []
            distances = []
            for i in range(X.shape[0]):
                forest.add_item(i, X[i])
            forest.build(10)

            n_neighbors = min(N - 1, 10 * self.perplexity)
            if isinstance(constraint_X, np.ndarray):
                constraint_N = constraint_X.shape[0]
                n_neighbors = min(n_neighbors, constraint_N - 1)
                if constraint_weight is None:
                    constraint_weight = np.ones(constraint_N)
                if constraint_labels is None:
                    constraint_labels = np.ones(constraint_N) * -1
            else:
                constraint_N = 0
                
            for i in range(X.shape[0]):
                ret = forest.get_nns_by_item(i, n_neighbors + 1, include_distances = True)
                indices.append(ret[0][1:])
                if labels[i] == -1:
                    distances.append(ret[1][1:])
                else:
                    try:
                        dist = []
                        for j in range(1, len(ret[1])):
                            if labels[i] == labels[ret[0][j]]:
                                dist.append(ret[1][j] * label_alpha)
                            else:
                                dist.append(ret[1][j])
                        distances.append(dist)
                    except:
                        distances.append(ret[1][1:])

            indices = np.array(indices, copy=True).reshape(-1)
            distances = np.array(distances, copy=True).reshape(-1)

            cffi_indices = self.ffi.new('int[]', list(indices))
            cffi_distances = self.ffi.new('double[]', list(distances))

            if isinstance(constraint_X, np.ndarray):
                cffi_constraint_X = self.ffi.cast('double*', constraint_X.ctypes.data)
                cffi_constraint_Y = self.ffi.cast('double*', constraint_Y.ctypes.data)
            else:
                cffi_constraint_X = self.ffi.NULL
                cffi_constraint_Y = self.ffi.NULL

            if constraint_N > 0:
                constraint_forest = AnnoyIndex(constraint_X.shape[1], 'euclidean')
                constraint_indices = []
                constraint_distances = []
                for i in range(constraint_X.shape[0]):
                    constraint_forest.add_item(i, constraint_X[i])
                constraint_forest.build(10)
                for i in range(X.shape[0]):
                    ret = constraint_forest.get_nns_by_vector(X[i], n_neighbors + 1, include_distances = True)
                    constraint_indices.append(ret[0][1:])
                    if labels[i] == -1:
                        constraint_distances.append(ret[1][1:])
                    else:
                        dist = []
                        for j in range(1, len(ret[1])):
                            if labels[i] == constraint_labels[ret[0][j]]:
                                dist.append(ret[1][j] * label_alpha)
                            else:
                                dist.append(ret[1][j])
                        constraint_distances.append(dist)
                constraint_indices = np.array(constraint_indices, copy=True).reshape(-1)
                constraint_distances = np.array(constraint_distances, copy=True).reshape(-1)
                cffi_constraint_indices = self.ffi.new('int[]', list(constraint_indices))
                cffi_constraint_distances = self.ffi.new('double[]', list(constraint_distances))
            else:
                cffi_constraint_indices = self.ffi.NULL
                cffi_constraint_distances = self.ffi.NULL
            if isinstance(corrected_id, np.ndarray) and isinstance(corrected_weight, np.ndarray) and isinstance(init_id, np.ndarray):
                corrected_N = corrected_id.shape[0]
                init_N = init_id.shape[0]
                assert corrected_id.shape[0] == corrected_weight.shape[0], "shape[0] of corrected_id should be equal to shape[0] of corrected_weight."
                assert corrected_id.shape[0] <= N, "shape[0] of corrected_id should not be greater than N."
                cffi_corrected_id = self.ffi.cast('int*', corrected_id.ctypes.data)
                cffi_init_id = self.ffi.cast('int*', init_id.ctypes.data)
                cffi_corrected_weight = self.ffi.cast('double*', corrected_weight.ctypes.data)
            else:
                corrected_N = 0
                cffi_corrected_id = self.ffi.NULL
                init_N = 0
                cffi_init_id = self.ffi.NULL
                cffi_corrected_weight = self.ffi.NULL

            X = X.reshape(-1)
            Y = Y.reshape(-1)
            cffi_X = self.ffi.cast('double*', X.ctypes.data)
            cffi_Y = self.ffi.cast('double*', Y.ctypes.data)

            #layouts = np.zeros(n_samples * self.n_components * 50)

            cffi_random_state = 1
            if self.random_state != None:
                cffi_random_state = self.random_state
            if constraint_N > 0:
                cffi_constraint_weight = self.ffi.cast('double*', constraint_weight.ctypes.data)
            else:
                cffi_constraint_weight = self.ffi.NULL
            t = FuncThread(self.C.run_bhtsne, cffi_X, N + n_labels, D, cffi_Y, self.n_components,
                           cffi_constraint_X, cffi_constraint_Y, constraint_N, alpha,
                           self.perplexity, self.angle, self.n_jobs, self.n_iter,
                           cffi_random_state, cffi_verbose, self.accuracy,
                           self.early_exaggeration, self.learning_rate,
                           skip_num_points, self._EXPLORATION_N_ITER,
                           n_neighbors, cffi_indices, cffi_constraint_indices, cffi_distances,
                           cffi_constraint_distances, cffi_constraint_weight, prev_n)
            t.daemon = True
            t.start()

            while t.is_alive():
                t.join(timeout=1.0)
                sys.stdout.flush()
            ret = Y.reshape((N+n_labels, self.n_components))
            return ret[:N]
            #return ret[:N], (means, ret[N:], [(distribution[i]['n'] ** 0.66) for i in distribution])

    def _tsne(self, P, degrees_of_freedom, n_samples, random_state, X_embedded,
              neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5
        }
        obj_func = self._kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exageration parameter
        P *= self.early_exaggeration
        params, kl_divergence, it = self.gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = self.gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] Error after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def gradient_descent(self, objective, p0, it, n_iter, n_iter_check=1,
                         n_iter_without_progress=300, momentum=0.8,
                         learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7,
                         verbose=0, args=None, kwargs=None):
        """
        Batch gradient descent with momentum and individual gains.

        Parameters
        ----------
        objective : function name, we choose a function to compute a
            tuple of cost and gradient for a given parameter vector.
            When expensive to compute, the cost can optionally be None
            and can be computed every n_iter_check steps using the
            objective_error function.

        p0 : array-like, shape (n_params,)
            Initial parameter vector.

        it : int
            Current number of iterations (this function will be called more than
            once during the optimization).

        n_iter : int
            Maximum number of gradient descent iterations.

        n_iter_check : int
            Number of iterations before evaluating the global error. If the error
            is sufficiently low, we abort the optimization.

        n_iter_without_progress : int, optional (default: 300)
            Maximum number of iterations without progress before we abort the
            optimization.

        momentum : float, within (0.0, 1.0), optional (default: 0.8)
            The momentum generates a weight for previous gradients that decays
            exponentially.

        learning_rate : float, optional (default: 200.0)
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers.

        min_gain : float, optional (default: 0.01)
            Minimum individual gain for each parameter.

        min_grad_norm : float, optional (default: 1e-7)
            If the gradient norm is below this threshold, the optimization will
            be aborted.

        verbose : int, optional (default: 0)
            Verbosity level.

        args : sequence
            Arguments to pass to objective function.

        kwargs : dict
            Keyword arguments to pass to objective function.

        Returns
        -------
        p : array, shape (n_params,)
            Optimum parameters.

        error : float
            Optimum.

        i : int
            Last iteration.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it
        tic = time()
        for i in range(it, n_iter):
            print(i)
            error, grad = objective(p, *args, **kwargs)
            grad_norm = linalg.norm(grad)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if (i + 1) % n_iter_check == 0:
                # temp = p.reshape(args[2], args[3])
                # fig = plt.figure(figsize=(15, 8))
                # ax = plt.subplot(111)
                # plt.scatter(temp[:, 0], temp[:, 1])
                # for i in range(0, args[2] - 800):
                #     plt.text(temp[i, 0], temp[i, 1], str(i), family='serif', style='italic', ha='right', wrap=True)
                # plt.axis('tight')
                # plt.show()
                toc = time()
                duration = toc - tic
                tic = toc
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: error = %.7f,"
                          " gradient norm = %.7f"
                          " (%s iterations in %0.3fs)"
                          % (i + 1, error, grad_norm, n_iter_check, duration))

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: did not make any progress "
                              "during the last %d episodes. Finished."
                              % (i + 1, n_iter_without_progress))
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                              % (i + 1, grad_norm))
                    break

        return p, error, i

    def joint_probabilities_nn(self, distances_nn, neighbors_nn, perplexity, verbose, n_neighbors, N):
        t0 = time()
        cffi_neighbors_nn = self.ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = self.ffi.cast('double*', distances_nn.ctypes.data)

        cffi_verbose = 0
        if verbose:
            cffi_verbose = 1

        conditional_P = np.zeros((N, n_neighbors))
        cffi_P = self.ffi.cast('double*', conditional_P.ctypes.data)
        t = FuncThread(self.C.binary_search_perplexity, cffi_distances_nn, cffi_neighbors_nn,
                       N, n_neighbors, perplexity, cffi_verbose, cffi_P)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        assert np.all(np.isfinite(conditional_P)), \
            "All probabilities should be finite"

        # Symmetrize the joint probability distribution using sparse operations
        P = csr_matrix((conditional_P.ravel(), neighbors_nn.ravel(),
                        range(0, N * n_neighbors + 1, n_neighbors)),
                       shape=(N, N))
        P = P + P.T

        # Normalize the joint probability distribution
        sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
        P /= sum_P

        assert np.all(np.abs(P.data) <= 1.0)
        if verbose >= 2:
            duration = time() - t0
            print("[t-SNE] Computed conditional probabilities in {:.3f}s"
                  .format(duration))
        return P

    def joint_probabilities(self, distances, desired_perplexity, verbose, is_square_form=True):
        """Compute joint probabilities p_ij from distances.

        Parameters
        ----------
        distances : array, shape (n_samples * (n_samples-1) / 2,)
            Distances of samples are stored as condensed matrices, i.e.
            we omit the diagonal and duplicate entries and store everything
            in a one-dimensional array.

        desired_perplexity : float
            Desired perplexity of the joint probability distributions.

        verbose : int
            Verbosity level.

        Returns
        -------
        P : array, shape (n_samples * (n_samples-1) / 2,)
            Condensed joint probability matrix.
        """
        # Compute conditional probabilities such that they approximately match
        # the desired perplexity
        distances = distances.astype(np.float64, copy=False)

        cffi_distances = self.ffi.cast('double*', distances.ctypes.data)

        cffi_verbose = 0
        if verbose:
            cffi_verbose = 1
        N, D = distances.shape
        conditional_P = np.zeros((N, D))
        cffi_P = self.ffi.cast('double*', conditional_P.ctypes.data)
        np.savetxt("distances.txt", distances)
        t = FuncThread(self.C.binary_search_perplexity, cffi_distances, self.ffi.NULL,
                       N, D, desired_perplexity, cffi_verbose, cffi_P)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        P = conditional_P + conditional_P.T
        sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
        if is_square_form:
            P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
        else:
            P = np.maximum(P / sum_P, MACHINE_EPSILON)
        return P

    def _kl_divergence(self, params, P, degrees_of_freedom, n_samples, n_components,
                       skip_num_points=0):
        """t-SNE objective function: gradient of the KL divergence
        of p_ijs and q_ijs and the absolute error.

        Parameters
        ----------
        params : array, shape (n_params,)
            Unraveled embedding.

        P : array, shape (n_samples * (n_samples-1) / 2,)
            Condensed joint probability matrix.

        degrees_of_freedom : float
            Degrees of freedom of the Student's-t distribution.

        n_samples : int
            Number of samples.

        n_components : int
            Dimension of the embedded space.

        skip_num_points : int (optional, default:0)
            This does not compute the gradient for points with indices below
            `skip_num_points`. This is useful when computing transforms of new
            data where you'd like to keep the old data fixed.

        Returns
        -------
        kl_divergence : float
            Kullback-Leibler divergence of p_ij and q_ij.

        grad : array, shape (n_params,)
            Unraveled gradient of the Kullback-Leibler divergence with respect to
            the embedding.
        """
        X_embedded = params.reshape(n_samples, n_components)

        # Q is a heavy-tailed distribution: Student's t-distribution
        dist = pdist(X_embedded, "sqeuclidean")
        dist += 1.
        dist /= degrees_of_freedom
        dist **= (degrees_of_freedom + 1.0) / -2.0
        dist_sum = 2.0 * np.sum(dist)
        Q = np.maximum(dist / dist_sum, MACHINE_EPSILON)

        # Optimization trick below: np.dot(x, y) is faster than
        # np.sum(x * y) because it calls BLAS

        # Objective: C (Kullback-Leibler divergence of P and Q)
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

        # Gradient: dC/dY
        # pdist always returns double precision distances. Thus we need to take
        grad = np.zeros(shape=(n_samples, n_components))
        # grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
        PQd = squareform((P - Q) * dist)
        for i in range(skip_num_points, n_samples):
            grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                             X_embedded[i] - X_embedded)

        grad = grad.ravel()
        c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
        grad *= c

        return kl_divergence, grad

    def k_neighbors(self, X1, X2, n_neighbors, knn_tree, T, V, L):
        cffi_X1 = self.ffi.cast('double*', X1.ctypes.data)
        N1, D1 = X1.shape
        cffi_X2 = self.ffi.cast('double*', X2.ctypes.data)
        N2, D2 = X2.shape
        assert D1 == D2, 'X1 and X2 should have same dim.'
        neighbors_nn = np.zeros((N1, n_neighbors), dtype=np.int)
        distances_nn = np.zeros((N1, n_neighbors), dtype=np.float64)
        cffi_neighbors_nn = self.ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = self.ffi.cast('double*', distances_nn.ctypes.data)

        cffi_verbose = 0
        if self.verbose:
            cffi_verbose = 1

        t = FuncThread(self.C.k_neighbors, cffi_X1, N1, cffi_X2, N2, D1, n_neighbors, cffi_neighbors_nn,
                       cffi_distances_nn, int(T), int(V), int(L), int(knn_tree), cffi_verbose)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()
        return neighbors_nn, distances_nn

    def assign_weight(self, data, selection,
                      source_tree_sizes=None,
                      source_label_distribution=None,
                      source_indexes=None,
                      neighbor_size=1):
        assert isinstance(data, np.ndarray), 'source_data should be 2D array.'
        source_data = np.array(data[[not x for x in selection]])
        target_data = np.array(data[selection])
        source_number, source_D = source_data.shape
        target_number, target_D = target_data.shape
        assert source_D == target_D, 'source_data and target_data should be same dim.'
        assert source_number > 0, 'source_data should not be null.'
        assert target_number > 0, 'target_data should not be null.'
        # if source_weights is None:
        #     source_weights = np.ones((len(data), ))
        assert source_label_distribution is not None, "you should give a initial source_label_distribution"
        if source_indexes is None:
            source_indexes = np.array(range(len(data)))
        # weight = source_weights[selection]
        tree_size = source_tree_sizes[selection]

        label_distribution = source_label_distribution[selection]
        if neighbor_size > target_number:
            neighbor_size = target_number
        from sklearn.neighbors import BallTree
        tree = BallTree(target_data)
        distances_nn, neighbors_nn = tree.query(source_data, neighbor_size, return_distance=True)
        # from scripts import knn_g
        # neighbors_nn, distances_nn = knn_g.Knn(source_data, source_number, source_D, neighbor_size, 1, 5, source_number)
        # exit(1)
        # neighbors_nn, distances_nn = self.k_neighbors(source_data, target_data, neighbor_size, 2, 1, 5, target_number)

        source_index = source_indexes[[not x for x in selection]]
        target_index = source_indexes[selection]
        next = []
        for i in range(target_number):
            next.append([])
        source_order_2_total_order = []
        for i in range(len(selection)):
            if not selection[i]:
                source_order_2_total_order.append(i)
        for i in range(source_number):
            for j in neighbors_nn[i]:
                next[j].append(source_order_2_total_order[i])

        # distances_nn = 1 / (distances_nn + 0.00001)
        # sum_nn = distances_nn.sum(axis=1)
        # source_weight = source_weights[[not x for x in selection]]
        source_tree_size = source_tree_sizes[[not x for x in selection]]
        source_label_distribution = source_label_distribution[[not x for x in selection]]
        for i in range(source_number):
            # weight[neighbors_nn[i]] += distances_nn[i] * (source_weight[i] / sum_nn[i])
            tree_size[neighbors_nn[i]] += source_tree_size[i]
            label_distribution[neighbors_nn[i]] += source_label_distribution[i]
        label_entropy = np.zeros((target_number, )) + 0.0001
        normal_label_distribution = label_distribution.transpose() / label_distribution.sum(axis=1)
        for i in range(len(normal_label_distribution)):
            for j in range(target_number):
                if normal_label_distribution[i][j] > 0 and normal_label_distribution[i][j] != 1:
                    label_entropy[j] += -normal_label_distribution[i][j] * np.log2(normal_label_distribution[i][j])

        level_info = {
            # 'other_index': source_index,
            'selection_index': target_index,
            'next': np.array(next),
            # 'selection_weight': weight,
            'selection_tree_size': tree_size,
            'selection_label_entropy': label_entropy,
            'selection_label_distribution': label_distribution
        }

        return level_info

if __name__ == "__main__":
    import numpy as np
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_embedded = IncrementalTSNE(n_components=2).fit_transform(X)
    print(X_embedded)
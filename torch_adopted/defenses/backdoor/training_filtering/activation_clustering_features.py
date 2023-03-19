import torch
from sklearn.decomposition import *
from sklearn.cluster import KMeans, MiniBatchKMeans, OPTICS
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from typing import TYPE_CHECKING, Sequence, Iterator
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data

import sklearn.decomposition._fastica as fastica
import numpy as np
from tenacity import retry, stop_after_attempt, wait_none

def _retryable_ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.
    Used internally by FastICA --main loop
    """
    W = fastica._sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = fastica._sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        # np.einsum allows having the lowest memory footprint.
        # It is faster than np.diag(np.dot(W1, W.T)).
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        raise ValueError("FastICA did not converge. Consider increasing "
                            "tolerance or the maximum number of iterations.")

    return W, ii + 1

# override _ica_par to use retry
fastica._ica_par = _retryable_ica_par


class ActivationClustering():
    r"""Activation Clustering proposed by Bryant Chen
    from IBM Research in SafeAI@AAAI 2019.

    It is a training filtering backdoor defense
    that inherits :class:`trojanvision.defenses.TrainingFiltering`.

    Activation Clustering assumes in the target class,
    poisoned samples compose a separate cluster
    which is small or far from its own class center.

    The defense procedure is:

    * Get feature maps for samples
    * For samples from each class

        * Get dim-reduced feature maps for samples using
          :any:`sklearn.decomposition.FastICA` or
          :any:`sklearn.decomposition.PCA`.
        * Conduct clustering w.r.t. dim-reduced feature maps and get cluster classes for samples.
        * Detect poisoned cluster classes. All samples in that cluster are poisoned.
          Poisoned samples compose a small separate class.

    There are 4 different methods to detect poisoned cluster classes:

    * ``'size'``: The smallest cluster class.
    * ``'relative size'``: The small cluster classes whose proportion is smaller than :attr:`size_threshold`.
    * ``'silhouette_score'``: only detect poison clusters using ``'relative_size'``
      when clustering fits data well.
    * ``'distance'``: Poison clusters are far from their own class center,

    See Also:
        * Paper: `Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering`_
        * Other implementation: `IBM adversarial robustness toolbox (ART)`_ [`source code`_]

    Args:
        nb_clusters (int): Number of clusters. Defaults to ``2``.
        nb_dims (int): The reduced dimension of feature maps. Defaults to ``10``.
        reduce_method (str): The method to reduce dimension of feature maps. Defaults to ``'FastICA'``.
        cluster_analysis (str): The method chosen to detect poisoned cluster classes.
            Choose from ``['size', 'relative_size', 'distance', 'silhouette_score']``
            Defaults to ``'silhouette_score'``.

    Note:
        Clustering method is :any:`sklearn.cluster.KMeans`
        if ``self.defense_input_num=None`` (full training set)
        else :any:`sklearn.cluster.MiniBatchKMeans`

    .. _Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering:
        https://arxiv.org/abs/1811.03728
    .. _IBM adversarial robustness toolbox (ART):
        https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/defences/detector_poisoning.html#art.defences.detector.poison.ActivationDefence
    .. _source code:
        https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/defences/detector/poison/activation_defence.py
    """  # noqa: E501

    name: str = 'activation_clustering'

    def __init__(
        self,
        classifier: Callable[[torch.Tensor], int],
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        dataloader: torch.utils.data.DataLoader,
        nb_clusters: int = 2,
        nb_dims: int = 10,
        reduce_method: str = 'FastICA',
        clustering_method: str = 'KMeans',
        cluster_analysis: str = 'silhouette_score',
        silhouette_threshold: float = 0.12,
    ):
        """
        @param nb_clusters: number of clusters (default: 2)
        @param nb_dims: the reduced dimension of feature maps (default: 10)
        @param reduce_method: the method to reduce dimension of feature maps
        @param cluster_analysis: the method chosen to detect poisoned cluster classes
                ['size', 'relative_size', 'distance', 'silhouette_score']
        """
        # super().__init__(**kwargs)
        self.nb_clusters = nb_clusters
        self.nb_dims = nb_dims
        self.reduce_method = reduce_method
        self.cluster_analysis = cluster_analysis
        self.dataloader = dataloader
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.silhouette_threshold = silhouette_threshold

        self.projector_reserved = PCA(n_components=self.nb_dims)

        match self.reduce_method:
            case 'FastICA':
                self.projector = fastica.FastICA(n_components=self.nb_dims, whiten='unit-variance', max_iter=2000, tol=1e-5)
            case 'PCA':
                self.projector = PCA(n_components=self.nb_dims)
            case 'FA':
                self.projector = FactorAnalysis(n_components=self.nb_dims)
            case 'IncrementalPCA':
                self.projector = IncrementalPCA(n_components=self.nb_dims)
            case 'KernelPCA':
                self.projector = KernelPCA(n_components=self.nb_dims)
            case 'LatentDirichletAllocation':
                self.projector = LatentDirichletAllocation(n_components=self.nb_dims)
            case 'MiniBatchSparsePCA':
                self.projector = MiniBatchSparsePCA(n_components=self.nb_dims)
            case 'NMF':
                self.projector = NMF(n_components=self.nb_dims)
            case 'MiniBatchNMF':
                self.projector = MiniBatchNMF(n_components=self.nb_dims)
            case 'SparsePCA':
                self.projector = SparsePCA(n_components=self.nb_dims)
            case 'TruncatedSVD':
                self.projector = TruncatedSVD(n_components=self.nb_dims)
            case _:
                raise ValueError(self.reduce_method + ' dimensionality reduction method not supported.')
        match clustering_method:
            case 'KMeans':
                self.clusterer = KMeans(n_clusters=self.nb_clusters, n_init="auto")
            case 'OPTICS':
                self.clusterer = OPTICS(n_jobs=1)

    @retry(stop=stop_after_attempt(5), wait=wait_none())
    def _get_projector_value(self, fm: torch.Tensor):
        return torch.as_tensor(self.projector.fit_transform(fm.numpy()))

    def get_pred_labels(self) -> torch.Tensor:
        all_fm = []
        all_pred_label = []
        loader = self.dataloader
        labels = set()

        # classifier ensurance feature for each sample
        classifier_ensurance = []
        # for _input, _label in tqdm(loader, leave=False):
        for _input, _label in loader:
            labels.update([l.item() for l in _label])
            # this is model features
            fm = self.feature_extractor(_input.to(self.device))
            pred_probs = torch.functional.softmax(self.classifier(fm), dim=1)
            assert len(pred_label.shape) > 1, 'classifier should return value for each class'
            pred_label = torch.argmax(pred_probs, dim=1)

            classifier_ensurance.append(pred_probs.detach().cpu())
            # we use flatten because feature extractor can return non 1d feature maps
            all_fm.append(torch.flatten(fm.detach().cpu(), 1, -1))
            all_pred_label.append(pred_label.detach().cpu())

        classifier_ensurance = torch.cat(classifier_ensurance)
        all_fm = torch.cat(all_fm)
        all_pred_label = torch.cat(all_pred_label)
        assert all_pred_label.shape == (len(all_fm),), 'all_pred_label and all_fm should have same length'

        result = torch.zeros_like(all_pred_label, dtype=torch.bool)

        idx_list: list[torch.Tensor] = []
        reduced_fm_centers_list: list[torch.Tensor] = []
        kwargs_list: list[dict[str, torch.Tensor]] = []
        all_clusters = {}
        all_clusters_flatten = torch.empty(all_pred_label.shape, dtype=torch.int)
        all_sample_silhuette = np.empty(all_pred_label.shape)
        all_sample_distance_to_cluster_centroid = np.empty(all_pred_label.shape)
        all_sample_mean_distance_to_cluster_centroid_amoung_cluster = np.empty(all_pred_label.shape)
        all_sample_relative_cluster_size = np.empty(all_pred_label.shape)
        all_sample_activation_norm = np.empty(all_pred_label.shape)
        all_sample_min_distance_to_other_classes = np.empty(all_pred_label.shape)
        all_reduced_fm = torch.empty(all_pred_label.shape[0], self.nb_dims)
        for _class in tqdm(labels, leave=False):
        # for _class in labels:
            idx = all_pred_label == _class
            fm = all_fm[idx]
            try:
                reduced_fm = self._get_projector_value(fm)
            except Exception:
                # if we can't reduce the dimension, we just user PCA projector
                reduced_fm = torch.as_tensor(self.projector_reserved.fit_transform(fm.numpy()))
            all_reduced_fm[idx] = reduced_fm.clone().detach()
            cluster_class = torch.as_tensor(self.clusterer.fit_predict(reduced_fm))
            all_clusters_flatten[idx] = cluster_class.clone().detach()
            all_clusters[_class] = cluster_class.clone().detach()
            kwargs_list.append(dict(cluster_class=cluster_class, reduced_fm=reduced_fm))
            idx_list.append(idx)
            cluster_centroids = torch.stack([
                reduced_fm[cluster_class == i].mean(dim=0) for i in range(self.nb_clusters)
            ])
            reduced_fm_centers_list.append(reduced_fm.median(dim=0))
            all_sample_silhuette[idx] = silhouette_score(reduced_fm, cluster_class)

            all_sample_distance_to_cluster_centroid[idx] = np.stack([
                torch.norm(reduced_fm[cluster_class == i] - cluster_centroids[i], p=2, dim=1).numpy()
                for i in range(self.nb_clusters)
            ])
            # TODO: should we norm all_sample_distance_to_cluster_centroid and other features?

            # mean distance from each sample to its cluster centroid
            all_sample_mean_distance_to_cluster_centroid_amoung_cluster[idx] = np.array([
                all_sample_distance_to_cluster_centroid[idx][cluster_class == i].mean()
                for i in range(self.nb_clusters)
            ])

            all_sample_relative_cluster_size[idx] = np.array([
                (cluster_class == i).sum() / len(cluster_class)
                for i in range(self.nb_clusters)
            ])

            all_sample_activation_norm[idx] = np.array([
                torch.norm(fm[i], 2).item()
                for i in range(len(cluster_class))
            ])

        reduced_fm_centers = torch.stack(reduced_fm_centers_list)

        all_poison_clusters = {}
        # for _class in tqdm(labels, leave=False):
        for _class in labels:
            # calculate minimum amoung distances for each sample to other classes
            no_this_class_center_mask = torch.ones_like(reduced_fm_centers)
            no_this_class_center_mask[_class] = 0
            all_sample_min_distance_to_other_classes[idx_list[_class]] = np.min(
                torch.norm(all_reduced_fm[idx_list[_class]] - reduced_fm_centers[no_this_class_center_mask], p=2, dim=1).numpy(),
                axis=1
            )

            # kwargs = kwargs_list[_class]
            # idx = torch.arange(len(all_pred_label))[idx_list[_class]]
            # kwargs['reduced_fm_centers'] = reduced_fm_centers

            # poison_cluster_classes = analyze_func(_class=_class, idx=idx, silhouette_threshold=self.silhouette_threshold, **kwargs)
            # for poison_cluster_class in poison_cluster_classes:
            #     result[idx[kwargs['cluster_class'] == poison_cluster_class]] = True

            # all_poison_clusters[_class] = poison_cluster_classes

        self.all_clusters = all_clusters
        # self.all_poison_clusters = all_poison_clusters

        # all activation clustering features for each sample
        self.all_ac_features = {
            'classifier_ensurance': classifier_ensurance,
            'all_fm': all_fm,
            'all_reduced_fm': all_reduced_fm,
            'all_pred_label': all_pred_label,
            # NOTE: cluster index may repeat amoung different predicted labels
            # but it is different clusters
            'all_clusters': all_clusters_flatten,
            'all_sample_silhuette': all_sample_silhuette,
            'all_sample_distance_to_cluster_centroid': all_sample_distance_to_cluster_centroid,
            'all_sample_mean_distance_to_cluster_centroid_amoung_cluster': all_sample_mean_distance_to_cluster_centroid_amoung_cluster,
            'all_sample_relative_cluster_size': all_sample_relative_cluster_size,
            'all_sample_activation_norm': all_sample_activation_norm,
            'all_sample_min_distance_to_other_classes': all_sample_min_distance_to_other_classes,
        }
        return result

    def analyze_by_size(self, cluster_class: torch.Tensor, **kwargs) -> list[int]:
        r"""The smallest cluster.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(1)``
        """
        return [cluster_class.bincount(minlength=self.nb_clusters).argmin().item()]

    def analyze_by_relative_size(self, cluster_class: torch.Tensor,
                                 size_threshold: float = 0.35,
                                 **kwargs) -> list[int]:
        r"""Small clusters whose proportion is smaller than :attr:`size_threshold`.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            size_threshold (float): Defaults to ``0.35``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``
        """
        relative_size = cluster_class.bincount(minlength=self.nb_clusters) / len(cluster_class)
        return torch.arange(self.nb_clusters)[relative_size < size_threshold].tolist()

    def analyze_by_silhouette_score(self, cluster_class: torch.Tensor,
                                    reduced_fm: torch.Tensor,
                                    silhouette_threshold: float,
                                    **kwargs) -> list[int]:
        """Return :meth:`analyze_by_relative_size()`
        if :any:`sklearn.metrics.silhouette_score` is high,
        which means clustering fits data well.

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            reduced_fm (torch.Tensor): Dim-reduced feature map tensor
                with shape ``(N, self.nb_dims)``
            silhouette_threshold (float): The threshold to calculate
                :any:`sklearn.metrics.silhouette_score`.
                Defaults to ``0.1``.

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``

        """
        if silhouette_score(reduced_fm, cluster_class) > silhouette_threshold:
            return self.analyze_by_relative_size(cluster_class, **kwargs)
        return []

    def analyze_by_distance(self, cluster_class: torch.Tensor,
                            reduced_fm: torch.Tensor,
                            reduced_fm_centers: torch.Tensor,
                            _class: int,
                            **kwargs) -> list[int]:
        r"""

        Args:
            cluster_class (torch.Tensor): Clustering result tensor
                with shape ``(N)``.
            reduced_fm (torch.Tensor): Dim-reduced feature map tensor
                with shape ``(N, self.nb_dims)``
            reduced_fm_centers (torch.Tensor): The centers of dim-reduced feature map tensors in each class
                with shape ``(C, self.nb_dims)``

        Returns:
            list[int]: Predicted poison cluster classes list with shape ``(K)``
        """
        cluster_centers_list = []
        for _class in range(self.nb_clusters):
            cluster_centers_list.append(reduced_fm[cluster_class == _class].median(dim=0))
        cluster_centers = torch.stack(cluster_centers_list)  # (self.nb_clusters, self.nb_dims)
        # (self.nb_clusters, C, self.nb_dims)
        differences = cluster_centers.unsqueeze(1) - reduced_fm_centers.unsqueeze(0)
        distances: torch.Tensor = differences.norm(p=2, dim=2)  # (self.nb_clusters, C)
        return torch.arange(self.nb_clusters)[distances.argmin(dim=1) != _class].tolist()

    def get_filtered_data(self, idx):
        """Return dataloader that returns only data with given idx in given order"""
        j = 0
        for i, data in enumerate(self.dataloader):
            assert idx[j] >= i
            if idx[j] == i:
                yield data
                j += 1

    def analyze_by_model_train(self, cluster_class: torch.Tensor,
                               idx: torch.Tensor,
                            reduced_fm: torch.Tensor,
                            reduced_fm_centers: torch.Tensor,
                            _class: int, **kwargs):
        """Train model on one cluster and check on other cluster"""
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from .kmeans_gpu import kmeans
from sklearn.cluster import KMeans


def k_means(embedding, k, device="cpu", distance="euclidean", centers='k-means++'):
    """
    K-means algorithm
    :param embedding: embedding of clustering
    :param k: hyper-parameter in K-means
    :param y_true: ground truth
    :param device: device
    :returns acc, nmi, ari, f1, center:
    - acc
    - nmi
    - ari
    - f1
    - cluster centers
    """
    if device == "cpu":
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if isinstance(centers, str):
            model = KMeans(n_clusters=k, n_init=20)
        else:
            model = KMeans(n_clusters=k, init=centers, n_init=1)
        cluster_id = model.fit_predict(embedding)
        center = model.cluster_centers_
    if device == "gpu" or 'cuda' in device:
        # if embedding is not torch tensor, convert it to torch tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding).to("cuda")
        cluster_id, center, _ = kmeans(X=embedding, num_clusters=k, distance=distance, device="cuda")
        cluster_id = cluster_id.numpy()
    return cluster_id, center

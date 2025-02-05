# -*- coding: utf-8 -*-

import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn import metrics
import pdb


def modularity(adjacency, clusters):
	"""Computes graph modularity.

	Args:
		adjacency: Input graph in terms of its sparse adjacency matrix.
		clusters: An (n,) int cluster vector.

	Returns:
		The value of graph modularity.
		https://en.wikipedia.org/wiki/Modularity_(networks)
	"""
	degrees = adjacency.sum(axis=0)
	n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
	result = 0
	for cluster_id in np.unique(clusters):
		cluster_indices = np.where(clusters == cluster_id)[0]
		adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
		degrees_submatrix = degrees[cluster_indices]
		result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
	return result / n_edges


def conductance(adjacency, clusters):
	"""Computes graph conductance as in Yang & Leskovec (2012).

	Args:
		adjacency: Input graph in terms of its sparse adjacency matrix.
		clusters: An (n,) int cluster vector.

	Returns:
		The average conductance value of the graph clusters.
	"""
	inter = 0  # Number of inter-cluster edges.
	intra = 0  # Number of intra-cluster edges.
	cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
	for cluster_id in np.unique(clusters):
		cluster_indices[:] = 0
		cluster_indices[np.where(clusters == cluster_id)[0]] = 1
		adj_submatrix = adjacency[cluster_indices, :]
		inter += np.sum(adj_submatrix[:, cluster_indices])
		intra += np.sum(adj_submatrix[:, ~cluster_indices])
	return intra / (inter + intra)


def match_cluster(y_true, y_pred):
	y_true = y_true - np.min(y_true) 
	cluster_num = np.max(y_true) + 1
	assert cluster_num == np.unique(y_true).shape[0]

	cost = np.zeros((cluster_num, cluster_num), dtype=int)
	for i in range(cluster_num):
		cluster1 = y_true == i
		for j in range(cluster_num):
			cluster2 = y_pred == j
			cost[i, j] = np.sum(np.logical_and(cluster1, cluster2))
	cost = cost.__neg__().tolist()

	m = Munkres()
	indexes = m.compute(cost)

	new_predict = np.zeros(len(y_pred))
	for i in range(cluster_num):
		j = indexes[i][1]
		new_predict[y_pred == j] = i

	return new_predict


def old_match_cluster(y_true, y_pred):
	# use Munkres algorithm to match the cluster ids and labels
	y_true = y_true - np.min(y_true)
	l1 = list(set(y_true))
	num_class1 = len(l1)
	l2 = list(set(y_pred))
	num_class2 = len(l2)
	ind = 0
	if num_class1 != num_class2:
		for i in l1:
			if i in l2:
				pass
			else:
				y_pred[ind] = i
				ind += 1
	l2 = list(set(y_pred))
	numclass2 = len(l2)
	if num_class1 != numclass2:
		print('error')
		return
	cost = np.zeros((num_class1, numclass2), dtype=int)
	for i, c1 in enumerate(l1):
		mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
		for j, c2 in enumerate(l2):
			mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
			cost[i][j] = len(mps_d)
	m = Munkres()
	cost = cost.__neg__().tolist()
	indexes = m.compute(cost)
	new_predict = np.zeros(len(y_pred))
	for i, c in enumerate(l1):
		c2 = l2[indexes[i][1]]
		ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
		new_predict[ai] = c
	return new_predict


def graph_evaluation(adj, cluster):
	"""
	evaluate the clustering performance
	:param adj: the adjacency matrix
	:param cluster: the cluster vector
	:returns mod, con:
	- modularity
	- conductance
	"""
	mod = modularity(adj, cluster)
	con = conductance(adj, cluster)
	return mod, con


def print_eval(y_pred, y_true, A):
	# all cluster_ids, true_labels, A are numpy arrays
	nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
	ari = ari_score(y_true, y_pred)

	new_pred = match_cluster(y_true, y_pred)
	# print(new_pred)

	acc = accuracy_score(y_true, new_pred)
	f1 = f1_score(y_true, new_pred, average='macro')

	mod = modularity(A, y_pred)
	con = conductance(A, y_pred)

	print('acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f, mod: %.2f, con: %.2f' % (acc*100, nmi*100, ari*100, f1*100, mod*100, con*100))
	return [acc, nmi, ari, f1, mod, con]


def print_eval_simple(y_pred, y_true):
	# no modularity and conductance
	# no need to pass A

	# all cluster_ids, true_labels, A are numpy arrays
	nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
	ari = ari_score(y_true, y_pred)

	new_pred = match_cluster(y_true, y_pred)
	# print(new_pred)

	acc = accuracy_score(y_true, new_pred)
	f1 = f1_score(y_true, new_pred, average='macro')

	print('acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f' % (acc*100, nmi*100, ari*100, f1*100))
	return [acc, nmi, ari, f1]
# -*- coding: utf-8 -*-

import os
import sys
import torch
import logging
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, TUDataset, KarateClub, WikipediaNetwork, WebKB, AttributedGraphDataset
from ogb.nodeproppred import NodePropPredDataset



def load_graph_data(root_path=".", dataset_name="dblp", show_details=False):
    """
    load graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        stream=sys.stdout)
    dataset_path = root_path + dataset_name
    if os.path.exists(dataset_path):
        logging.info("Loading " + dataset_name + " dataset from local")
        load_path = root_path + dataset_name + "/" + dataset_name
        feat = np.load(load_path+"_feat.npy", allow_pickle=True)
        label = np.load(load_path+"_label.npy", allow_pickle=True)
        adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    else:
        logging.info("Loading " + dataset_name + " dataset from pyg")
        if dataset_name in ["chameleon", "crocodile", "squirrel"]:
            data = WikipediaNetwork(root=root_path+'pyg', name=dataset_name)
        elif dataset_name== "wisc":
            data = WebKB(root=root_path+'pyg', name="Wisconsin")
        elif dataset_name == 'flickr':
            data = AttributedGraphDataset(root=root_path+'pyg', name='Flickr')
        elif dataset_name == 'blogcatalog':
            data = AttributedGraphDataset(root=root_path+'pyg', name='BlogCatalog')
        elif dataset_name == 'pubmed':
            data = Planetoid(root=root_path+'pyg', name='PubMed')
        elif dataset_name == 'arxiv':
            data = NodePropPredDataset(name='ogbn-arxiv', root=root_path+'pyg')
        else:
            raise NotImplementedError("The dataset is not supported")
        data = data[0]

        if dataset_name == 'arxiv':
            graph, label = data
            label = label.flatten()
            feat = graph['node_feat']
            adj_idx = graph['edge_index']
        else:
            feat = data.x.to_dense().numpy()
            label = data.y.numpy()
            adj_idx = data.edge_index.numpy()
        adj_sp = sp.coo_matrix((np.ones(adj_idx.shape[1]), (adj_idx[0], adj_idx[1])), shape=(feat.shape[0], feat.shape[0]))
        adj = adj_sp.toarray()



    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:   ", int(adj.sum()/2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


def load_data(root_path="./", dataset_name="USPS", show_details=False):
    """
    load non-graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - category num
    - category distribution
    :returns feat, label: the features and labels
    """
    root_path = root_path + "dataset/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        pass
    load_path = root_path + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("------details of dataset------")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("category num:   ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label


    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """

    data = WikipediaNetwork(root='dataset/pyg', name=dataset_name)
    data = data[0]
    feat = data.x.numpy()
    label = data.y.numpy()
    adj_idx = data.edge_index.numpy()
    adj_sp = sp.coo_matrix((np.ones(adj_idx.shape[1]), (adj_idx[0], adj_idx[1])), shape=(feat.shape[0], feat.shape[0]))
    adj = adj_sp.toarray()


    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:       ", int(adj.sum() / 2))
        print("category num:   ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj
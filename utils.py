import torch
import torch.nn.functional as F
import numpy as np
import pdb


def cluster_id2assignment(cluster_ids, cluster_num):
    C = torch.zeros(cluster_ids.shape[0], cluster_num)
    C[torch.arange(cluster_ids.shape[0]), cluster_ids] = 1
    return C


def Cprop(C0, A, args):
    C = C0
    for _ in range(args.cprop_layers):
        C = args.cprop_alpha * torch.spmm(A, C) + C0
    return C


def t(C0, A, args):
    C = C0
    for _ in range(args.low_pass_layers):
        C = args.alphaC * torch.spmm(A, C) + C0
    return C


def hetero_deg(A, label):
    edge_index = np.array(A.nonzero())
    return (label[edge_index[0]] == label[edge_index[1]]).sum()/edge_index.shape[1]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(label, num_classes, percls_trn, val_lb, seed=42):
    num_nodes = label.shape[0]
    index=[i for i in range(num_nodes)]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(label.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=num_nodes)
    val_mask = index_to_mask(val_idx,size=num_nodes)
    test_mask = index_to_mask(test_idx,size=num_nodes)
    return train_mask, val_mask, test_mask
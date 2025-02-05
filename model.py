from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch
from dgc.clustering import k_means
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch_geometric.nn import DMoNPooling, GCNConv

from torch_geometric.nn.models.mlp import MLP
from typing import Optional, Tuple
from torch_geometric.nn import Linear
from math import log
from torch.nn import ModuleList
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import MLP


from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers=2):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(n_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.last_conv = GCNConv(hidden_channels, hidden_channels)
        

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x).relu()
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_conv(x, edge_index, edge_weight)
        return x


class encoding(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, low_pass_filter):
        super(encoding, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.low_pass_filter = low_pass_filter
        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        if args.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(2*hidden_dim, 2*hidden_dim))
            # self.last_layer = nn.Linear(2*hidden_dim, output_dim)
        elif args.fusion_method == 'add' or args.fusion_method == 'max':
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # self.last_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise NotImplementedError
    
        

    def construct_high_pass_adj(self, H_0, adj):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (adj!=0)
        return F.relu(pair_wise_cos)
    

    def fusion(self, H_low, H_high):
        if self.args.fusion_method == 'add':
            H =  self.args.fusion_beta * H_low + (1-self.args.fusion_beta) * H_high
        elif self.args.fusion_method == 'concat':
            H = torch.cat((self.args.fusion_beta*H_low, (1-self.args.fusion_beta)*H_high), dim=1)
        elif self.args.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        return H
    

    def forward(self, X, adj):
        H_0 = self.first_layer(X)
        self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0, adj))
        self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 l=self.args.high_pass_layers, 
                                                 alpha=self.args.high_pass_alpha)

        self.H_low = self.low_pass_fc(self.low_pass_filter @ H_0)
        self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        H = self.fusion(self.H_low, self.H_high)        

        H = H + self.args.fusion_gamma * self.fusion_fc(H)
        # H = self.last_layer(H)
        # H = F.softmax(H, dim=1)
        
        return H


class encoding2(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, norm_adj, cluster_num):
        super(encoding2, self).__init__()
        self.args = args

        if args.first_transformation=='mlp':
            self.first_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        elif args.first_transformation=='linear':
            self.first_layer = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        
        self.norm_adj = norm_adj
        self.cluster_num = cluster_num

        self.lins = ModuleList()
        for i in range(self.args.low_pass_layers):
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))

        self.low_pass_fc  = nn.Linear(hidden_dim, hidden_dim)
        self.high_pass_fc = nn.Linear(hidden_dim, hidden_dim)

        self.final_layer = Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot')

        self.cluster_layer = Linear(in_channels = hidden_dim, out_channels = self.cluster_num, bias = True, weight_initializer = 'glorot')
        #nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                             # nn.ReLU(),
                                             # nn.Linear(hidden_dim, self.cluster_num))

        if args.fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*hidden_dim, 2*hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(2*hidden_dim, 2*hidden_dim))
            # self.last_layer = nn.Linear(2*hidden_dim, output_dim)
        elif args.fusion_method == 'add' or args.fusion_method == 'max':
            self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
            # self.fusion_param = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
            # self.last_layer = nn.Linear(hidden_dim, output_dim)
        else:
            raise NotImplementedError
    
        

    def construct_high_pass_adj(self, H_0):
        H_0_norm = F.normalize(H_0, p=2, dim=1)
        pair_wise_cos = torch.mm(H_0_norm, H_0_norm.t())
        pair_wise_cos = pair_wise_cos * (self.norm_adj!=0)
        return F.relu(pair_wise_cos)
    

    def fusion(self, H_low, H_high):
        if self.args.fusion_method == 'add':
            H =  self.args.fusion_beta * H_low + (1-self.args.fusion_beta) * H_high
        elif self.args.fusion_method == 'concat':
            H = torch.cat((self.args.fusion_beta*H_low, (1-self.args.fusion_beta)*H_high), dim=1)
        elif self.args.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        return H
    

    def forward(self, X, mask: Optional[Tensor] = None):
        H_0 = self.first_layer(X)
        # self.high_pass_adj = normalize_adj_torch(self.construct_high_pass_adj(H_0))
        # self.high_pass_filter = construct_filter(adj=self.high_pass_adj, 
                                                 # l=self.args.high_pass_layers, 
                                                 # alpha=self.args.high_pass_alpha)
        H = H_0
        for i in range(self.args.low_pass_layers):
            # temp = torch.mm(H_0.t(), H)
            # temp = torch.mm(H_0, temp)
            # DeProp(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)

            H = self.args.alphaH * torch.spmm(self.norm_adj, H) + H_0
            H = (1-self.args.alphaH) * torch.spmm(self.norm_adj, H) + self.args.alphaH * H_0
            H = DePropagate(H, H_0, self.norm_adj, self.args.step_size_gamma, self.args.alphaH, self.args.alphaO)
            # H = F.normalize(H, p=2, dim=1)



            # H_low = self.alpha*self.norm_adj(H_low) + H_0

            # theta = log(self.args.gamma / (i + 2))
            # H = (1 - theta) * H + theta * self.lins[i].forward(H)
            # H = H + self.lins[i].forward(H)
            # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
            # H = F.relu(H, inplace = True)
            # H = F.selu(H, inplace = True)

        # self.H_high = self.high_pass_fc(self.high_pass_filter @ H_0)
        # H = self.fusion(self.H_low, self.H_high)        

        # H = H + self.args.fusion_gamma * self.fusion_fc(H)
        H = self.final_layer(H)
        # H = F.dropout(H, p = self.args.dropout, training = self.training, inplace = True)
        H = F.normalize(H, p=2, dim=1)

        
        return H



class top_agg(nn.Module):
    def __init__(self, A_norm, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(top_agg, self).__init__()

        if linear_prop=='sgc':
            I = torch.eye(A_norm.shape[0]).to(A_norm.device)
            top_filter = torch.eye(A_norm.shape[0]).to(A_norm.device)
            for _ in range(hop):
                top_filter = alpha * A_norm @ top_filter + I
            self.top_filter = top_filter
            if linear_trans=='lin':
                self.fc = nn.Linear(emb_dim, emb_dim)
            elif linear_trans=='mlp':
                self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
        elif linear_prop=='gcn':
            self.top_filter = A_norm
            self.lins = ModuleList()
            self.lins.append(Linear(in_channels = emb_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            for i in range(hop-2):
                self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = emb_dim, bias = True, weight_initializer = 'glorot'))
        else:
            raise NotImplementedError
        self.linear_prop = linear_prop
        self.norm = norm
        

    def agg(self, x):
        return self.top_filter @ x

    def forward(self, x):
        if self.linear_prop == 'sgc':
            x = self.fc(x)
            # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
            if self.norm == 'l2-norm':
                print('l2-norm')
                x = F.normalize(x, p=2, dim=1)
        elif self.linear_prop == 'gcn':
            for lin in self.lins[:-1]:
                x = lin(self.top_filter @ x)
                x = F.relu(x)
            x = self.lins[-1](self.top_filter @ x)
        
        
        return x



class top_agg_f(nn.Module):
    # the diffusion is computed during the forward pass 
    # A_norm is given in sparse format
    # gcn branch is removed for simplicity
    def __init__(self, A_norm, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(top_agg_f, self).__init__()

        self.A_norm = A_norm
        self.alpha = alpha
        self.hop = hop

        if linear_trans=='lin':
            self.fc = nn.Linear(emb_dim, emb_dim)
        elif linear_trans=='mlp':
            self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)

        self.linear_prop = linear_prop
        self.norm = norm

    def _top_filter(self, x):
        top_filter = x
        for _ in range(self.hop):
            top_filter = self.alpha * torch.sparse.mm(self.A_norm, top_filter) + x
        return top_filter

    def forward(self, x):
        x = self._top_filter(x)
        x = self.fc(x)
        # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
        if self.norm == 'l2-norm':
            print('l2-norm')
            x = F.normalize(x, p=2, dim=1)

        
        
        return x






class top_agg2(nn.Module):
    def __init__(self, A_norm, alpha, hop, emb_dim, hidden_dim, linear_trans='lin'):
        super(top_agg2, self).__init__()

        self._top_filter(A_norm, alpha, hop)

        if linear_trans=='lin':
            self.fc = nn.Linear(emb_dim, emb_dim)
        elif linear_trans=='mlp':
            self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
    
    
    def _top_filter(self, A_norm, alpha, hop):
        I = torch.eye(A_norm.shape[0]).to(A_norm.device)
        top_filter = torch.eye(A_norm.shape[0]).to(A_norm.device)
        for _ in range(hop):
            top_filter = alpha * A_norm @ top_filter + I
        self.top_filter = top_filter

    def agg(self, x):
        return self.top_filter @ x

    def forward(self, x):
        x = self.fc(x)
        
        return x







def compute_attr_simi_mtx(X, attr_r, bin=0):
    ### contruct the attribute similarity matrix ###
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    attr_simi_mtx[attr_simi_mtx<0] = 0
    # attr_simi_mtx = attr_simi_mtx - torch.diag_embed(torch.diag(attr_simi_mtx))

    row, col = torch.nonzero(attr_simi_mtx, as_tuple=True)
    values = attr_simi_mtx[row, col] 
    _, sort_indices = torch.sort(values, descending=True)

    keep_size = int(attr_r * len(sort_indices))
    sort_indices = sort_indices[:keep_size]

    values = values[sort_indices]
    if bin == 1:
        values = torch.ones_like(values)
    row = row[sort_indices]
    col = col[sort_indices]
    attr_simi_mtx = torch.sparse_coo_tensor(torch.stack((row, col),0), values, attr_simi_mtx.shape)
    attr_simi_mtx = attr_simi_mtx.to_dense()

    return attr_simi_mtx


def compute_gaussian_kernel_mtx(X, sigma=0.2):
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    attr_simi_mtx = torch.exp(-attr_simi_mtx/(2 * sigma ** 2))
    return attr_simi_mtx


def compute_knn_simi_mtx(X, k=10):
    X_n = F.normalize(X, p=2, dim=1)
    attr_simi_mtx = (X_n@X_n.t()).to(X.device)
    _, indices = torch.topk(attr_simi_mtx, k, dim=1)
    row = torch.arange(attr_simi_mtx.shape[0]).repeat(k, 1).t().reshape(-1).to(X.device)
    col = indices.reshape(-1).to(X.device)
    values = torch.ones_like(row).float().to(X.device)
    attr_simi_mtx = torch.sparse_coo_tensor(torch.stack((row, col),0), values, attr_simi_mtx.shape)
    attr_simi_mtx = attr_simi_mtx.to_dense()
    return attr_simi_mtx



class attr_agg(nn.Module):
    def __init__(self, attr_simi_mtx, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(attr_agg, self).__init__()

        self.attr_simi_mtx = attr_simi_mtx

        ### setup the linear propagation model ###
        if linear_prop=='sgc':
            I = torch.eye(attr_simi_mtx.shape[0]).to(attr_simi_mtx.device)
            attr_filter = torch.eye(attr_simi_mtx.shape[0]).to(attr_simi_mtx.device)
            for _ in range(hop):
                attr_filter = alpha * attr_simi_mtx @ attr_filter + I
            self.attr_filter = attr_filter
            if linear_trans=='lin':
                self.fc = nn.Linear(emb_dim, emb_dim)
            elif linear_trans=='mlp':
                self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)
        elif linear_prop=='gcn':
            self.attr_filter = attr_simi_mtx
            self.lins = ModuleList()
            self.lins.append(Linear(in_channels = emb_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            for i in range(hop-2):
                self.lins.append(Linear(in_channels = hidden_dim, out_channels = hidden_dim, bias = True, weight_initializer = 'glorot'))
            self.lins.append(Linear(in_channels = hidden_dim, out_channels = emb_dim, bias = True, weight_initializer = 'glorot'))
        else:
            raise NotImplementedError
        self.linear_prop = linear_prop
        self.norm = norm

    def agg(self, x):
        return self.attr_filter @ x

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1)
        if self.linear_prop == 'sgc':
            x = self.fc(x)
            # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
            if self.norm == 'l2-norm':
                print('l2-norm')
                x = F.normalize(x, p=2, dim=1)
        elif self.linear_prop == 'gcn':
            for lin in self.lins[:-1]:
                x = lin(self.attr_filter @ x)
                x = F.relu(x)
            x = self.lins[-1](self.attr_filter @ x)
        
        return x




class attr_agg_f(nn.Module):
    def __init__(self, half_S_norm, alpha, hop, emb_dim, hidden_dim, linear_prop='sgc', linear_trans='lin', norm=None):
        super(attr_agg_f, self).__init__()
        ### setup the linear propagation model ###
        self.half_S_norm = half_S_norm
        self.alpha = alpha
        self.hop = hop

        if linear_trans=='lin':
            self.fc = nn.Linear(emb_dim, emb_dim)
        elif linear_trans=='mlp':
            self.fc = MLP(in_channels=emb_dim, hidden_channels=hidden_dim, out_channels=emb_dim, num_layers=2, batch_norm=False, dropout=0.0, bias=True)

        self.linear_prop = linear_prop
        self.norm = norm


    def _top_filter(self, x):
        attr_filter = x
        for _ in range(self.hop):
            temp = self.half_S_norm.t() @ attr_filter
            attr_filter = self.alpha * self.half_S_norm @ temp + x
        return attr_filter
    

    def forward(self, x):
        # x = F.normalize(x, p=2, dim=1)
        x = self._top_filter(x)

        x = self.fc(x)
        # x = (x - x.mean(0)) / x.std(0) / torch.sqrt(torch.tensor(x.shape[1]).to(x.device))
        if self.norm == 'l2-norm':
            print('l2-norm')
            x = F.normalize(x, p=2, dim=1)

        
        return x





class fusion(nn.Module):
    def __init__(self, fusion_method, fusion_beta, emb_dim, fusion_norm=None):
        super(fusion, self).__init__()
        self.fusion_method = fusion_method
        self.fusion_beta = fusion_beta
        self.fusion_norm = fusion_norm
        if fusion_method == 'concat':
            self.fusion_fc = nn.Linear(2*emb_dim, emb_dim)
        else: 
            self.fusion_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, H_low, H_high, beta=None): # an in-place normalization
        if beta is None:
            beta = self.fusion_beta

        H_low = beta * H_low
        H_high = (1-beta) * H_high

        if self.fusion_method == 'add':
            H =  H_low + H_high
        elif self.fusion_method == 'concat':
            H = self.fusion_fc(torch.cat((H_low, H_high), dim=1))
            H = F.normalize(H, p=2, dim=1) 
        elif self.fusion_method == 'max':
            H = torch.max(H_low, H_high)
        else:
            raise NotImplementedError
        
        return H



class C_agg(nn.Module):
    def __init__(self, alpha, hop, A):
        super(C_agg, self).__init__()

        I = torch.eye(A.shape[0]).to(A.device)
        C_filter = torch.eye(A.shape[0]).to(A.device)
        for _ in range(hop):
            C_filter = alpha * torch.spmm(A, C_filter) + I
        # C_filter -= I
        self.C_filter = C_filter

        
    def forward(self, C):
        return self.C_filter @ C
    


class C_agg_f(nn.Module):
    def __init__(self, alpha, hop, A):
        super(C_agg_f, self).__init__()

        self.alpha = alpha
        self.hop = hop
        self.A = A

        
    def forward(self, C):
        C_filter = C
        for _ in range(self.hop):
            C_filter = self.alpha * torch.spmm(self.A, C_filter) + C
        return C_filter




class attr_agg2(nn.Module):
    def __init__(self, X, alpha, hop, input_dim, hidden_dim):
        super(attr_agg2, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        return self.fc(x)

class hetero_agg(nn.Module):
    def __init__(self, alpha, hop, A):
        super(hetero_agg, self).__init__()

        self.alpha = alpha
        self.hop = hop
        self.A = A

    def forward(self, C, H):
        A_hetero = self.A - C@C.t()
        A_hetero = F.relu(A_hetero)
        A_hetero = (A_hetero + A_hetero.t())/2

        L_hetero = torch.eye(A_hetero.shape[0]).to(C.device)
        L_hetero = L_hetero * A_hetero.sum(axis=1) - A_hetero

        I = torch.eye(L_hetero.shape[0]).to(C.device)
        filter_hetero = torch.eye(L_hetero.shape[0]).to(C.device)
        for _ in range(self.hop):
            filter_hetero = self.alpha * filter_hetero @ L_hetero + I
        H = F.normalize(filter_hetero @ H, p=2, dim=1)
        return H


    

class cluster_model(nn.Module):
    def __init__(self, method, node_num, hidden_dim, cluster_num, H):
        super(cluster_model, self).__init__()
        self.method = method
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num

        if method == 'mlp':
            self.fc = nn.Linear(hidden_dim, cluster_num).to(H.device)
            # self.cluster_assi = F.softmax(self.fc(H), dim=1)
        elif method == 'kmeans':
            init = self.kmeans(H)
            self.cluster_assi = nn.Parameter(init)
        elif method == 'random':
            init = torch.rand(node_num, cluster_num)
            self.cluster_assi = nn.Parameter(init)
        else:
            raise NotImplementedError
        
          
    def kmeans(self, H):
        cluster_ids,_ = k_means(H, self.cluster_num, device='gpu', distance='cosine')
        cluster = torch.zeros(H.shape[0], self.cluster_num).to(H.device)
        cluster[torch.arange(H.shape[0]), cluster_ids] = 1
        return cluster
    

    def forward(self, H):
        if self.method == 'mlp':
            return F.softmax(self.fc(H), dim=1)
        else:
            return F.softmax(self.cluster_assi, dim=1)


class low_pass_model(nn.Module):
    # H_0 = MLP(X) or H_0 = linear(X)
    # H = \sum_{i=1}^{L} \alpha_i A^i H_0 W
    def __init__(self, args, low_pass_filter, input_dim, hidden_dim):
        super(low_pass_model, self).__init__()
        self.args = args
        self.low_pass_filter = low_pass_filter
        if args.first_transformation=='mlp':
            self.lin1 = MLP([input_dim, hidden_dim])
        elif args.first_transformation=='linear':
            self.lin1 = nn.Linear(input_dim, hidden_dim)
        else:
            raise NotImplementedError
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        H_0 = self.lin1(X)
        H = self.low_pass_filter @ H_0
        H = self.lin2(H)
        return H



class pool_based_model(nn.Module):
    def __init__(self, encoder, input_dim, hidden_dim, cluster_num, dropout, low_pass_filter, args):
        super(pool_based_model, self).__init__()
        self.encoder = encoder
        if encoder == 'GCN':
            self.encoding = GCNConv(input_dim, hidden_dim)
        elif encoder == 'low_pass':
            self.encoding = low_pass_model(args, low_pass_filter, input_dim, hidden_dim)
        else:
            raise NotImplementedError    
        self.dmon_pool = DMoNPooling(hidden_dim, cluster_num, dropout=dropout)
    
    def forward(self, X, A, edge_index):
        if self.encoder == 'GCN':
            H = self.encoding(X, edge_index)
        elif self.encoder == 'low_pass':
            H = self.encoding(X)
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon_pool(H, A)
        # remove the dimension for graph level
        return s[0], out[0], out_adj[0], spectral_loss, ortho_loss, cluster_loss


def SSG_CCA_loss_fn(H1, H2):
    loss1 = F.mse_loss(H1, H2)
    # loss2 = F.mse_loss(H1.t() @ H1, torch.eye(H1.shape[1]).to(H1.device))
    # loss3 = F.mse_loss(H2.t() @ H2, torch.eye(H2.shape[1]).to(H2.device))
    # loss = loss1 + 0.1*(loss2 + loss3)
    return loss1

def ortho_loss_fn(H):
    return F.mse_loss(H.t() @ H, torch.eye(H.shape[1]).to(H.device))

def node_t_neighbor_a_loss_fn(H_t, H_a, A):
    H_a = A @ H_a
    loss = F.mse_loss(H_t, H_a)
    return loss

def node_t_neighbor_a_loss_fn2(H_t, H_a, A_no_loop_sym):
    # A_norm = normalize_adj_torch(A_ori, self_loop=False, symmetry=False)
    H_a = torch.spmm(A_no_loop_sym, H_a)
    loss = F.mse_loss(H_t, H_a)
    # loss = (H_t - H_a).norm(p=2, dim=1).mean()

    return loss

def node_t_cluster_a_loss_fn(H_t, H_a, C, simi=None, centers=None):
    if simi is None:
        simi = torch.ones(H_t.shape[0]).to(H_t.device)
    if centers is None:
        C = F.normalize(C, p=2, dim=1)
        centers = C.t() @ H_a # K x d
    predict_labels = torch.argmax(C, dim=1)
    centers = centers[predict_labels]
    # loss = torch.pow((simi * (H_t - centers).norm(p=2, dim=1)), 2).mean()
    loss = (simi * (H_t - centers).norm(p=2, dim=1)).mean()
    return loss


def node_t_cluster_a_loss_fn2(H_t, H_a, C, simi=None, centers=None, clu_size=True):
    if centers is None:
        # print(C.sum(0), C.sum(0).mean())
        if clu_size == False:
            print('clu_size is False')
            C = F.normalize(C, p=1, dim=0)
        # C = C/C.sum(0).mean()
        centers = C.t() @ H_a # K x d
    predict_labels = torch.argmax(C, dim=1)
    centers = centers[predict_labels]
    if simi is None:
        loss = F.mse_loss(H_t, centers)
        # loss = (H_t - centers).norm(p=2, dim=1).mean()
        # loss = (H_t - centers).pow(2).sum(dim=1).mean()
    else:
        loss = (simi * (H_t - centers).pow(2)).mean()
    return loss


def kmeans_loss_fn(H, C, args):
    if args.kmeans_loss == 'tr':
        return node_t_cluster_a_loss_fn(H, H, C)
    elif args.kmeans_loss == 'cen':
        return kmeans_centroid_contrastive_loss_fn(H, C, args)
    elif args.kmeans_loss == 'nod':
        return kmeans_node_contrastive_loss_fn(H, C)
    else:
        raise NotImplementedError


def kmeans_trace_loss_fn(H, C, centers=None):
    if centers is None:
        centers = C.T @ H
    loss = 0
    for i in range(C.shape[1]):
        loss += torch.pow(H[torch.argmax(C, dim=1) == i] - centers[i],2).sum(axis=1).mean()
        # loss += torch.pow(((H - centers[i]) * C[:, i].unsqueeze(1)),2).sum()
    return loss/C.shape[0]/C.shape[1]


def kmeans_centroid_contrastive_loss_fn(H, C, args):
    # pos: node representation vs. its cluster centroid
    # neg: node representation vs. other cluster centroids


    # C_bin = torch.zeros_like(C)
    # pred_clu = torch.argmax(C, dim=1)
    # C_bin[torch.arange(C.shape[0]), pred_clu] = 1
    # C_bin = F.normalize(C_bin, p=1, dim=0)
    # cluster_centroid_embedding = C_bin.T @ H
    C = F.normalize(C, p=1, dim=0)
    cluster_centroid_embedding = C.T @ H

    H = F.normalize(H, p=2, dim=1)
    cluster_centroid_embedding = F.normalize(cluster_centroid_embedding, p=2, dim=1)
    sim = torch.einsum("nd, kd -> nk", H, cluster_centroid_embedding)
    sim = sim * C
    sim /= args.temperature
    labels = torch.argmax(C, dim=1)
    return F.cross_entropy(sim, labels) 

# TODO: implement this function
def density_estimation(H, C, args):
    # concentration estimation (phi)        
    Dcluster = []
    for i in range(C.shape[1]):
        Dcluster.append((H[torch.argmax(C, dim=1)==i] - H[torch.argmax(C, dim=1)==i].mean(1)).norm(p=2, dim=1).list())
    
    
    density = np.zeros(C.shape[1])
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = args.temperature*density/density.mean()  #scale the mean to temperature 

def kmeans_node_contrastive_loss_fn(H, C, num_neg_samples):
    # pos: nodes in the same cluster
    # neg: nodes in different clusters

    labels = torch.argmax(C, dim=1)
    num_nodes = H.shape[0]
    loss = 0

    for i in range(num_nodes):
        target_label = labels[i]
        pos_indices = torch.where(labels == target_label)[0]
        neg_indices = torch.where(labels != target_label)[0]

        # 随机选择一个正样本
        pos_index = pos_indices[torch.randperm(pos_indices.size(0))[0]]
        pos_sample = H[pos_index]

        # 随机选择 num_neg_samples 个负样本
        neg_indices = neg_indices[torch.randperm(neg_indices.size(0))[:num_neg_samples]]
        neg_samples = H[neg_indices]

        # 计算 InfoNCE loss
        pos_sim = torch.dot(H[i], pos_sample)
        neg_sim = torch.matmul(H[i], neg_samples.T)
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=H.device)
        loss += F.cross_entropy(logits, targets)

    return loss / num_nodes


def spectral_loss_fn(C, A):
    temp = torch.matmul(C.t(), A)
    return -torch.trace(torch.matmul(temp, C))


def reg_loss_fn(C, args):
    if args.reg_loss == 'orth':
        # || C^T C - I ||_2^F
        return torch.norm(torch.matmul(C.t(), C)-
                      torch.eye(C.shape[1]).to(C.device), p=2)
    elif args.reg_loss == 'col':
        # sqrt(K)/n ||sum C_i ||_F - 1
        return torch.norm(C.sum(dim=0), p=2) * np.sqrt(C.shape[1])/C.shape[0] -1
    elif args.reg_loss == 'sqrt':
        # -trace(sqrt(C^T C))
        return -torch.trace(torch.sqrt(torch.matmul(C.t(), C)+1e-15))
    else:
        raise NotImplementedError
    


def reconstruction_loss_distance(A, B):
    return torch.norm(A-B, p=2)


def reconstruction_loss_cosine(A, B, beta=1):
    A = F.normalize(A, p=2, dim=1)
    B = F.normalize(B, p=2, dim=1)
    return torch.pow(1 - torch.sum(A*B, dim=1), beta).sum()


def DePropagate(C, C0, A, gamma, alphaC, alphaO):
    z =  (1 - gamma * alphaC + gamma * alphaO) * C
    s = gamma * alphaC * torch.spmm(A, C)
    t = gamma * alphaO * C @ (C.t() @ C)
    return z + s - t + gamma*C0



class input_enc(nn.Module):
    """preprocess S or A, used as input to following model and attr_model
    args:
        model (str): svd on S or A, lin(ear) on S or A, or mlp on S or A
        dims (list): [hidden_dim] for svd, [input_dim, hidden_dim] for lin, [input_dim, hidden_dim1, hidden_dim2, ..., hidden_dimn] for mlp
    """
    def __init__(self, enc, input_dim, hidden_dim, emb_dim):
        super(input_enc, self).__init__()
        self.model = enc
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        if self.model == 'lin':
            self.fc = nn.Linear(input_dim, emb_dim)
        elif self.model == 'mlp':
            self.fc = MLP([input_dim, hidden_dim, emb_dim], batch_norm=False, dropout=0.0, bias=True)
    

    def init_U(self, mtx1, mtx2):
        if self.model == 'svd':
            U, _, _ = torch.svd_lowrank(mtx1, q=self.emb_dim, niter=7)
            self.U = U
        elif self.model == 'lin' or self.model == 'mlp':
            self.U = mtx2
        
        

    def forward(self):
        if self.model == 'svd':
            return self.U
        else:
            return self.fc(self.U)



class pre_process_x(nn.Module):
    """preprocess X, used in the reconstruction loss
    args: 
        model (str): svd on smoothed X, lin(ear) on smoothed X, or mlp on smoothed X
        dims (list): [hidden_dim] for svd, [input_dim, hidden_dim] for lin, [input_dim, hidden_dim1, hidden_dim2, ..., hidden_dimn] for mlp
    """
    def __init__(self, model, dims):

        super(pre_process_x, self).__init__()
        self.model = model
        self.dims = dims
        if model == 'lin':
            self.fc = nn.Linear(dims[0], dims[1])
        elif model == 'mlp':
            self.fc = MLP(dims, batch_norm=False, dropout=0.0, bias=True)
    def forward(self, x):
        if self.model == 'svd':
            U, s, _ = torch.svd_lowrank(x, q=self.dims[0], niter=7)
            return U @ torch.diag(s)
        else:
            return self.fc(x)
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
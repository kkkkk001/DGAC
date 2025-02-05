import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, pdb, sys, argparse
from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch, normalize_adj_torch_sparse
import scipy.sparse as sp
from model import *
from dgc.clustering import k_means
from dgc.eval import print_eval, match_cluster, print_eval_simple
from dgc.rand import setup_seed
from datetime import datetime
import time
from distutils.util import strtobool
import sys
# sys.path.insert(0, './DeProp')
# from DeProp.model import DeProp
setup_seed(42)
torch.autograd.set_detect_anomaly(True)
start_time = time.time()
print('start time:', datetime.now())
from utils import cluster_id2assignment, Cprop
cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.nn.models.mlp import MLP



def spectral_clu(H, cluster_num):
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=cluster_num, assign_labels='kmeans').fit(H)
    cluster_id = clustering.labels_
    cluster_centers = np.zeros((cluster_num, H.shape[1]))
    for i in range(cluster_num):
        cluster_centers[i] = H[cluster_id==i].mean(0)
    return cluster_id, cluster_centers



parser = argparse.ArgumentParser(description='my_model')
### training params ###
parser.add_argument('--dataset', type=str, default='cora', help="name of dataset")
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--t_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--t_wd', type=float, default=5e-4, help='')
parser.add_argument('--a_wd', type=float, default=5e-4, help='')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--dropout', type=float, default=0.0, help='')
parser.add_argument('--fold', type=str, default='0-1-2-3-4', help='num of repeats, and seed of each repeat, separated by -')


parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dimension")
parser.add_argument('--emb_dim', type=int, default=64, help="hidden dimension")


parser.add_argument('--input_encoder', type=str, default='svd', help='svd, lin, or mlp')
parser.add_argument('--svd_on_S', type=str, default='S_norm', help='obtain input features by svd on S or S_norm')
parser.add_argument('--svd_on_A', type=str, default='A_norm', help='obtain input features by svd on S or A_norm')


### top_agg params ###
parser.add_argument('--top_layers', type=int, default=5, help='')
parser.add_argument('--top_alpha', type=float, default=0.5, help='')
parser.add_argument('--top_prop', type=str, default='sgc', help='sgc style, gcn style, appnp style, or deprop style')
parser.add_argument('--top_linear_trans', type=str, default='lin', help='lin or mlp')



### attr_agg params ###
parser.add_argument('--attr_layers', type=int, default=5, help='')
parser.add_argument('--attr_alpha', type=float, default=0.5, help='')
parser.add_argument('--attr_r', type=float, default=0.5, help='the nnz ratio of attr simi mtx')
parser.add_argument('--attr_bin', type=int, default=0, help='the nnz ratio of attr simi mtx')
parser.add_argument('--attr_prop', type=str, default='sgc', help='sgc style, gcn style, or appnp style')
parser.add_argument('--attr_linear_trans', type=str, default='lin', help='lin or mlp')



### fusion params ###
parser.add_argument('--fusion_norm', type=str, default='none', help='if l2-norm, l2 normalization on Ht and HA before fusion, if none, no post process')
parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='H = beta * H_t + (1-beta) * H_a')


### X_prop params ###
parser.add_argument('--xprop_layers', type=int, default=5, help='')
parser.add_argument('--xprop_alpha', type=float, default=0.5, help='')

### C_prop params ###
parser.add_argument('--cprop_layers', type=int, default=5, help='')
parser.add_argument('--cprop_alpha', type=float, default=0.2, help='')
parser.add_argument('--cprop_abl', type=int, default=0, help='')


### loss params ###
parser.add_argument('--loss_lambda_prop', type=float, default=1, help='')
parser.add_argument('--sharpening', type=float, default=1, help="")
parser.add_argument('--loss_lambda_kmeans', type=float, default=0.1, help='')
parser.add_argument('--kmeans_loss', type=str, default='tr', 
                    help='tr(ace), cen(troid contrastive), nod(e contrastive)')
parser.add_argument('--loss_lambda_SSG0', type=float, default=0.001, help='')
parser.add_argument('--loss_lambda_SSG1', type=float, default=0.01, help='')
parser.add_argument('--loss_lambda_SSG2', type=float, default=0.01, help='')
parser.add_argument('--loss_lambda_SSG3', type=float, default=0.01, help='')
parser.add_argument('--temperature', type=float, default=1, help='') 
parser.add_argument('--clu_size', type=strtobool, default=True, help='') 
parser.add_argument('--norm', type=int, default=0, help='')
parser.add_argument('--rounding', type=int, default=0, help='')


### log params ###
parser.add_argument('--log_file', type=str, default=None, help='')
parser.add_argument('--log_fold_file', type=str, default=None, help='')
parser.add_argument('--log_title', type=strtobool, default=True, help='')
parser.add_argument('--save_model', type=strtobool, default=False, help='')


parser.add_argument('--clu_check', type=int, default=1, help='')


args = parser.parse_args()
if args.log_file is None:
    args.log_file = 'res_log/'+args.dataset+'_grid_search.txt'
if args.log_fold_file is None:
    args.log_fold_file = 'res_fold_log/'+args.dataset+'_grid_search.txt'

print(args)
args_keys = []
args_values = []
for key, value in vars(args).items():
    args_keys.append(key)
    args_values.append(str(value))

##### load data #####
X, true_labels, A = load_graph_data(root_path='./dataset/', 
                          dataset_name=args.dataset, show_details=True)
cluster_num = len(np.unique(true_labels))


edge_index = torch.LongTensor(np.array(A.nonzero())).to(args.device)
# in sparse version
A = torch.FloatTensor(A).to_sparse().to(args.device)
A_norm = normalize_adj_torch_sparse(A, self_loop=True, symmetry=True)
A_no_loop_sym = normalize_adj_torch_sparse(A, self_loop=False, symmetry=False)

X = torch.FloatTensor(X).to(args.device)
X_norm = F.normalize(X, p=2, dim=1)

# compute half_S_norm
# such that half_S_norm @ half_S_norm.t() = S_norm
deg_vec = X_norm @ X_norm.sum(0)
deg_vec[deg_vec == 0] = 1
deg_vec = deg_vec.pow(-0.5)
half_S_norm = deg_vec.unsqueeze(1) * X_norm
# 不完全相等，有精度差累积



def train():
    ##### train #####
    best_acc = 0
    best_res = []

    best_loss = 99999999

    beta = args.fusion_beta
    for e in range(args.epochs):
        optimizer_t.zero_grad()
        optimizer_a.zero_grad()

        H_t = model(S_encoder())
        H_a = attr_model(A_encoder())
        H = fusion_attr(H_t, H_a)


        # check if H contains nan
        if torch.isnan(H).sum() > 0:
            print('nan in H')
            pdb.set_trace()


        loss_prop = torch.pow(1 - cos_sim(H, X_prop), args.sharpening).mean()

        if e%args.clu_check == 0:
            if args.rounding == 0:
                if e == 0:
                    cluster_ids,centers_xprop = k_means(X_prop.detach().cpu(), cluster_num, device='cpu', distance='cosine')
                    cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers=centers_xprop)
                else:
                    cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')
            else:
                print('rounding')
                cluster_ids,centers = k_means((torch.round(torch.tanh(H)*7)).detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')


        C0 = cluster_id2assignment(cluster_ids, cluster_num).to(args.device)
        # C = Cprop(C0, A, args)
        if args.cprop_abl == 1:
            print('cprop_abl')
            C = C0
        else:
            C = C_prop_model(C0)
        # C = C0
        C = F.normalize(C, p=2, dim=1)

        loss_kmeans =  kmeans_loss_fn(H, C, args)
        


        if args.norm == 1:
            print('norm')
            H_t_norm = (H_t - H_t.mean(dim=0)) / H_t.std(dim=0) / torch.sqrt(torch.tensor(H_t.shape[1]).to(H_t.device))
            H_a_norm = (H_a - H_a.mean(dim=0)) / H_a.std(dim=0) / torch.sqrt(torch.tensor(H_a.shape[1]).to(H_a.device))
        elif args.norm == 0:
            H_t_norm = H_t
            H_a_norm = H_a
        ort_loss = (ortho_loss_fn(H_t_norm) + ortho_loss_fn(H_a_norm))
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).pow(2).sum()
        inv_loss_o = F.mse_loss(H_t_norm, H_a_norm)
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).norm(p=2, dim=1).mean()
        inv_loss_n = (node_t_neighbor_a_loss_fn2(H_t_norm, H_a_norm, A_no_loop_sym) + node_t_neighbor_a_loss_fn2(H_a_norm, H_t_norm, A_no_loop_sym))
        # print(C.sum(0).mean(), C.sum(0).max())
        inv_loss_c = (node_t_cluster_a_loss_fn2(H_t_norm, H_a_norm, C, clu_size=args.clu_size) + node_t_cluster_a_loss_fn2(H_a_norm, H_t_norm, C, clu_size=args.clu_size))
        inv_loss = args.loss_lambda_SSG1 * inv_loss_o + args.loss_lambda_SSG2 * inv_loss_n + args.loss_lambda_SSG3 * inv_loss_c


        loss = args.loss_lambda_prop * loss_prop + args.loss_lambda_kmeans * loss_kmeans + args.loss_lambda_SSG0 * ort_loss + inv_loss
        loss.backward()
        optimizer_t.step()
        optimizer_a.step()


        ## evaluation
        print('epoch: %d , loss: %.3f, loss_kmeans: %.3f, loss_prop: %.3f, ort_loss: %.3f,' % (e, loss.item(), loss_kmeans.item(), loss_prop, ort_loss), end=' ')
        print('inv_loss1: %.3f, inv_loss2: %.3f, inv_loss3: %.3f,' % (inv_loss_o, inv_loss_n, inv_loss_c), end=' ')
        predict_labels = torch.argmax(C, dim=1)
        res = print_eval_simple(predict_labels.cpu().numpy(), true_labels)
        

        ## best acc
        if res[0] > best_acc:
            best_acc = res[0]
            best_res = res
            best_e = e
            if args.save_model:
                torch.save(H.cpu().detach(), best_model_path+'H.pth')
        if loss < best_loss:
            best_loss = loss
            if args.save_model:
                torch.save(H.cpu().detach(), best_model_path+'best_loss_H.pth')

    print(f'best epoch: {best_e}')
    return best_res




retain_grah=True 
total_res = []
# for fold in range(5):
# final result is the mean of 5 repeated experiments
for fold in [int(x) for x in args.fold.split('-')]:
    setup_seed(43+fold)
    print("#"*60)
    print("#"*26, ' fold:%d ' % fold, "#"*26)
    print("#"*60)


    ##### compute low rank US_norm and UA_norm #####
    S_encoder = input_enc(args.input_encoder, X.shape[0], args.hidden_dim, args.emb_dim).to(args.device)
    A_encoder = input_enc(args.input_encoder, X.shape[0], args.hidden_dim, args.emb_dim).to(args.device)
    # store init US_norm / UA_norm within the model 
    # for svd, use the first mtx; for lin or mlp, use the second mtx
    if args.input_encoder == 'svd':
        pre_saved_path = '/home/kxie/cluster/dataset/pre_saved_U/'+args.dataset+'/'
        if os.path.exists(pre_saved_path+args.svd_on_S+'.pth'):
            print('load pre-saved U')
            S_encoder.U = torch.load(pre_saved_path+args.svd_on_S+'.pth').to(args.device)
            A_encoder.U = torch.load(pre_saved_path+args.svd_on_A+'.pth').to(args.device)
        else:
            print('compute U')
            os.makedirs(pre_saved_path, exist_ok=True)
            X_cpu = X.cpu()
            S = compute_attr_simi_mtx(X_cpu, args.attr_r, args.attr_bin)
            S_norm = normalize_adj_torch(S, self_loop=False, symmetry=True)
            S_encoder.init_U(eval(args.svd_on_S), None) 
            A_encoder.init_U(eval(args.svd_on_A), None)
            S_encoder.U = S_encoder.U.to(args.device)
            A_encoder.U = A_encoder.U.to(args.device)
            print('save U')
            torch.save(S_encoder.U, pre_saved_path+args.svd_on_S+'.pth')
            torch.save(A_encoder.U, pre_saved_path+args.svd_on_A+'.pth')
    else:
        S_encoder.init_U(None, X@X.t()) 
        A_encoder.init_U(None, A)
        # S_encoder.U = S_encoder.U.to(args.device)
        # A_encoder.U = A_encoder.U.to(args.device)

    #### define model and optimizer #####
    model = top_agg_f(A_norm, args.top_alpha, args.top_layers, args.emb_dim, args.hidden_dim, linear_prop=args.top_prop, linear_trans=args.top_linear_trans, norm=args.fusion_norm).to(args.device)
    attr_model = attr_agg_f(half_S_norm, args.attr_alpha, args.attr_layers, args.emb_dim, args.hidden_dim, linear_prop=args.attr_prop, linear_trans=args.attr_linear_trans, norm=args.fusion_norm).to(args.device)

    fusion_attr = fusion(args.fusion_method, args.fusion_beta, args.hidden_dim).to(args.device)
    
    C_prop_model = C_agg_f(args.cprop_alpha, args.cprop_layers, A_norm).to(args.device)


    optimizer_t = torch.optim.Adam(list(model.parameters())+list(S_encoder.parameters()), lr=args.t_lr, weight_decay=args.t_wd)
    optimizer_a = torch.optim.Adam(list(attr_model.parameters())+list(A_encoder.parameters()), lr=args.a_lr, weight_decay=args.a_wd)


    ##### compute low rank X_prop #####
    X_prop = X_norm
    for _ in range(args.xprop_layers):
        X_prop = args.xprop_alpha * torch.spmm(A_norm, X_prop) + X_norm
    U, s, _ = torch.svd_lowrank(X_prop, q=args.emb_dim, niter=7)
    X_prop = U @ torch.diag(s)
    X_prop = F.normalize(X_prop, p=2, dim=1)


    ##### training and evaluation #####
    best_model_path = f'./best_model/{args.dataset}/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    best_model_path += f'fold{fold}_'
    res = train()
    total_res.append(res)

    print('fold: %d, acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f' % (fold, res[0]*100, res[1]*100, res[2]*100, res[3]*100))


total_res = np.array(total_res)

print('***** final result: *****')
print('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n'%(args.dataset, 
    total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
    total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
    total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
    total_res[:, 3].mean()*100, total_res[:, 3].std()*100))

print(total_res)


total_time = time.time() - start_time
print('total time:', total_time)


# save the result
with open(args.log_file, 'a+') as f:
    # if the file is empty, write the header
    if args.log_title:
        f.write('input_encoder, top_linear_trans, fusion_norm, contras_norm, clu_size,')
        f.write('layers, attr_layers, alpha, S_alpha, ssg0, ssg1, ssg2, ssg3, fusion_beta')
        f.write('dataset, acc_mean, acc_std, nmi_mean, nmi_std, ari_mean, ari_std, f1_mean, f1_std, mod_mean, mod_std, con_mean, con_std, time, ')
        f.write(', '.join(x for x in args_keys)+'\n')
    f.write(', '.join([str(x) for x in [args.input_encoder, args.top_linear_trans, args.fusion_norm, args.norm, args.clu_size]])+', ')
    f.write(', '.join([str(x) for x in [args.top_layers, args.attr_layers, args.top_alpha, args.attr_alpha, args.loss_lambda_SSG0, args.loss_lambda_SSG1, args.loss_lambda_SSG2, args.loss_lambda_SSG3, args.fusion_beta]])+', ')
    f.write('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, '%(args.dataset, 
        total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
        total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
        total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
        total_res[:, 3].mean()*100, total_res[:, 3].std()*100, total_time))
    f.write(', '.join(x for x in args_values)+'\n')
    

# save the result of each fold
with open(args.log_fold_file, 'a+') as f:
    # if the file is empty, write the header
    if args.log_title:
        f.write('input_encoder, top_linear_trans, fusion_norm, contras_norm, clu_size,')
        f.write('layers, attr_layers, alpha, S_alpha, ssg0, ssg1, ssg2, ssg3, fusion_beta')
        f.write('dataset, fold, acc_mean, nmi_mean, ari_mean, f1_mean, mod_mean, con_mean, time, ')
        f.write(', '.join(x for x in args_keys)+'\n')
    for i in range(total_res.shape[0]):
        f.write(', '.join([str(x) for x in [args.top_layers, args.attr_layers, args.top_alpha, args.attr_alpha, args.loss_lambda_SSG0, args.loss_lambda_SSG1, args.loss_lambda_SSG2, args.loss_lambda_SSG3, args.fusion_beta]])+', ')
        f.write(', '.join([str(x) for x in [args.input_encoder, args.top_linear_trans, args.fusion_norm, args.norm, args.clu_size]])+', ')
        f.write('%s, %d, %.2f, %.2f, %.2f, %.2f, %.2f, '%(args.dataset, i, 
            total_res[i, 0]*100, total_res[i, 1]*100, total_res[i, 2]*100, total_res[i, 3]*100, total_time))
        f.write(', '.join(x for x in args_values)+'\n')



print('end time:', datetime.now())



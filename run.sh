python train.py --dataset texas --epochs 500 --svd_on_A A --fusion_norm none --norm 1 \
--top_layers 5 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 5 --attr_alpha 0.5 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.5 \
--xprop_layers 5 --xprop_alpha 0.2 --cprop_layers 1 --cprop_alpha 1 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.006  --loss_lambda_SSG2 0.006 --loss_lambda_SSG3 0.006 



python train.py --dataset wisc --epochs 500 --fusion_norm none --norm 1  \
--top_layers 5 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 5 --attr_alpha 0.2 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.3 \
--xprop_layers 5 --xprop_alpha 0.2 --cprop_layers 6 --cprop_alpha 0.2 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.001 --loss_lambda_SSG1 0.006  --loss_lambda_SSG2 0.006 --loss_lambda_SSG3 0.006



python train.py --dataset cornell --epochs 500 --fusion_norm l2-norm --norm 1 \
--top_layers 4 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 4 --attr_alpha 0.5 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.5 \
--xprop_layers 4 --xprop_alpha 0.2 --cprop_layers 2 --cprop_alpha 0.2 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.001 --loss_lambda_SSG1 0.006  --loss_lambda_SSG2 0.006 --loss_lambda_SSG3 0.006 



python train.py --dataset squirrel --epochs 500 --input_encoder mlp --fusion_norm none --norm 1 \
--top_layers 1 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 1 --attr_alpha 0.2 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.9 \
--xprop_layers 1 --xprop_alpha 0.2 --cprop_layers 1 --cprop_alpha 0.2 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0.0001 --loss_lambda_SSG3 0.0001



python train.py --dataset chameleon  --epochs 500 --input_encoder mlp --fusion_norm l2-norm --norm 1 \
--top_layers 1 --top_alpha 0.2 --top_prop sgc --top_linear_trans mlp \
--attr_layers 1 --attr_alpha 0.2 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.8 \
--xprop_layers 1 --xprop_alpha 0.2 --cprop_layers 1 --cprop_alpha 0.2 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0.0001 --loss_lambda_SSG3 0.0001 



python train.py --dataset flickr  --epochs 500 --input_encoder mlp --fusion_norm l2-norm --norm 1 --clu_size False \
--top_layers 1 --top_alpha 0.8 --top_prop sgc --top_linear_trans mlp \
--attr_layers 1 --attr_alpha 0.8 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.1 \
--xprop_layers 1 --xprop_alpha 0.8 --cprop_layers 1 --cprop_alpha 0.1 \
--kmeans_loss cen --loss_lambda_kmeans 0.006 --temperature 4 --loss_lambda_prop 5 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0.0001 --loss_lambda_SSG3 0.0001 






python train.py --dataset cora --epochs 500 --input_encoder svd --fusion_norm l2-norm --norm 0 \
--top_layers 8 --top_alpha 1.0 --top_prop sgc --top_linear_trans lin \
--attr_layers 8 --attr_alpha 1.0 --attr_r 1 --attr_prop sgc --attr_linear_trans lin \
--fusion_method add --fusion_beta 0.8 \
--xprop_layers 8 --xprop_alpha 1.0 --cprop_layers 5 --cprop_alpha 1.0 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.001 --loss_lambda_SSG1 0.00001 --loss_lambda_SSG2 0.00001 --loss_lambda_SSG3 0.00001 



python train.py --dataset citeseer  --epochs 500 --input_encoder svd --fusion_norm none --norm 0 \
--top_layers 5 --top_alpha 0.8 --top_prop sgc --top_linear_trans lin \
--attr_layers 5 --attr_alpha 0.8 --attr_r 1 --attr_prop sgc --attr_linear_trans lin \
--fusion_method add --fusion_beta 0.5 \
--xprop_layers 5 --xprop_alpha 0.8 --cprop_layers 5 --cprop_alpha 0.8 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 1 --sharpening 1 \
--loss_lambda_SSG0 0.001 --loss_lambda_SSG1 0.00001  --loss_lambda_SSG2 0.00001 --loss_lambda_SSG3 0.00001



python train.py --dataset pubmed  --epochs 500 --input_encoder svd --fusion_norm none --norm 1 --clu_size False \
--top_layers 1 --top_alpha 0.8 --top_prop sgc --top_linear_trans lin \
--attr_layers 1 --attr_alpha 0.2 --attr_r 1 --attr_prop sgc --attr_linear_trans lin \
--fusion_method add --fusion_beta 0.5 \
--xprop_layers 1 --xprop_alpha 0.8 --cprop_layers 1 --cprop_alpha 0.8 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 2 --sharpening 1 \
--loss_lambda_SSG0 0 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0.0001 --loss_lambda_SSG3 0.0001 



python train.py --dataset blogcatalog  --epochs 500 --input_encoder mlp --fusion_norm l2-norm --norm 1 --clu_size False \
--top_layers 3 --top_alpha 0.8 --top_prop sgc --top_linear_trans mlp \
--attr_layers 3 --attr_alpha 0.8 --attr_r 1 --attr_prop sgc --attr_linear_trans mlp \
--fusion_method add --fusion_beta 0.1 \
--xprop_layers 3 --xprop_alpha 0.8 --cprop_layers 3 --cprop_alpha 0.1 \
--kmeans_loss cen --loss_lambda_kmeans 0.006 --temperature 4 --loss_lambda_prop 5 --sharpening 1 \
--loss_lambda_SSG0 0.0001 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0.0001 --loss_lambda_SSG3 0.0001 



python train.py --dataset bat  --epochs 500 --input_encoder lin --fusion_norm none --norm 1 --t_wd 0 --a_wd 0 --t_lr 1e-3 --a_lr 1e-3 \
--top_layers 3 --top_alpha 1 --top_prop sgc --top_linear_trans lin \
--attr_layers 1 --attr_alpha 1 --attr_r 1 --attr_prop sgc --attr_linear_trans lin \
--fusion_method add --fusion_beta 1 \
--xprop_layers 3 --xprop_alpha 1 --cprop_layers 3 --cprop_alpha 1 \
--kmeans_loss cen --loss_lambda_kmeans 0 --temperature 2 --loss_lambda_prop 0 --sharpening 1 \
--loss_lambda_SSG0 0 --loss_lambda_SSG1 0.001  --loss_lambda_SSG2 0 --loss_lambda_SSG3 0.0001 --clu_size False 



python train.py --dataset uat  --epochs 500 --input_encoder lin --fusion_norm none --norm 1 --t_wd 0 --a_wd 0 --t_lr 1e-3 --a_lr 1e-3 \
--top_layers 3 --top_alpha 1 --top_prop sgc --top_linear_trans lin \
--attr_layers 1 --attr_alpha 1 --attr_r 1 --attr_prop sgc --attr_linear_trans lin \
--fusion_method add --fusion_beta 1 \
--xprop_layers 3 --xprop_alpha 1 --cprop_layers 3 --cprop_alpha 1 \
--kmeans_loss cen --loss_lambda_kmeans 0.02 --temperature 2 --loss_lambda_prop 0 --sharpening 1 \
--loss_lambda_SSG0 0 --loss_lambda_SSG1 0.0001  --loss_lambda_SSG2 0 --loss_lambda_SSG3 0.0001 --clu_size False --rounding 1


## Diffusion-based Graph-agnostic Clustering

### Test environment
The test environment are recorded in *environment.yml*. We recommond install with *conda* from the environment file. 

### Datasets 
|  | # nodes | # edges | # attributes | # clusters | Homo. Ratio |
|---|---|---|---|---|---|
| Texas | 183 | 325 | 1703 | 5 | 0.108 |
| Wisconsin | 251 | 515 | 1703 | 5 | 0.196 |
| Cornell | 183 | 298 | 1703 | 5 | 0.305 |
| Squirrel | 5201 | 108536 | 2089 | 5 | 0.223 |
| Chameleon | 2277 | 18050 | 2325 | 5 | 0.235 |
| Flickr | 7575 | 239738 | 12047 | 9 | 0.239 |
| Cora | 2708 | 5429 | 1433 | 7 | 0.81 |
| Citeseer | 3327 | 4732 | 3703 | 6 | 0.739 |
| Pubmed | 19717 | 44324 | 500 | 3 | 0.771 |
| BlogCatalog | 5196 | 171743 | 8189 | 6 | 0.40 |
| BAT | 131 | 1038 | 81 | 4 | 0.45 |
| UAT | 1190 | 13599 | 239 | 4 | 0.698 |

We have include Texas, Cornell, Cora, Citeseer, BAT and UAT datasets under *dataset/* directory for test. Other datasets can be loaded from torch_geometric. 


### Running code
To reproduce the results in the paper, please run the provided script.
```
bash run.sh
```

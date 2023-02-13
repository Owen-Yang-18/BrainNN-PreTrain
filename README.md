# Code repository for CHIL'23 submission "Pre-training Graph Neural Networks for Brain Network Analysis".
We will be updating the full implementation pipeline after acceptance. Thanks!
## Dataset
The raw file for [PPMI](https://www.ppmi-info.org/) dataset used for pre-training can be accessed in the `Data/` folder
## Parameters
1. Run the main file will start the pre-training process, all parameters are defaulted to following:
- **backbone**, default=(GCN, GAT, GIN). The backbone encoder.
- **rdim**, type=`int`. The dimension the atlas mapping pre-preocessing reduce to.
- **filename**, type=`str`. The filename used to store the pre-trained parameters. Must be using the suffix `.pth`.
## Requirements (latest versions recommended)
- torch
- numpy
- scipy
- higher
- torch_geometric
- networkx
- random
- sklearn

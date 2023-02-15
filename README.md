# Code repository for CHIL'23 submission "Pre-training Graph Neural Networks for Brain Network Analysis".
We will be updating the full implementation pipeline after acceptance. Thanks!
## Dataset
The raw file for [PPMI](https://www.ppmi-info.org/) dataset used for pre-training can be accessed in the `Data/` folder. The `.mat` file stores the adjacency connections of smapled brain networks from **three** different tractography algorithms (aka. views). The `.xlsx` file stores the atlas template of **Desikan-Killiany 84** parcellation system, and is further processed/dumped into a numpy `.arr` file for faster look-up.
## Parameters
1. Run the main file will start the pre-training process, all parameters are defaulted to following:
- **backbone**, default = (GCN, GAT, GIN). The backbone encoder.
- **rdim**, type = `int`. The dimension the atlas mapping pre-preocessing reduce to.
- **filename**, type = `str`. The filename used to store the pre-trained parameters. Must be using the suffix `.pth`.
## Requirements (latest versions recommended)
- torch
- numpy
- scipy >= 1.8.0
- higher
- torch_geometric
- networkx < 2.7.0
- random
- sklearn

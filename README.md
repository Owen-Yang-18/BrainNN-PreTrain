# Pre-Training and Meta-Learning of Graph Neural Networks for Brain Network Analysis
<kbd> <img src="https://github.com/Owen-Yang-18/BrainNN-PreTrain/blob/main/Figures/meta.png"> </kbd>

<kbd> <img src="https://github.com/Owen-Yang-18/BrainNN-PreTrain/blob/main/Figures/pipeline.png"> </kbd>

This repository contains the implementation of the frameworks presented in the [PTGB: Pre-Train Graph Neural Networks for Brain Network Analysis (CHIL 2023 **Oral**)](https://arxiv.org/pdf/2305.14376.pdf) and the [Data-Efficient Brain Connectome Analysis via Multi-Task Meta-Learning (SIGKDD 2022)](https://arxiv.org/pdf/2206.04486.pdf). The figures above present the visual illustrations for multi-task GNN meta-learning (top) and self-supervised pre-training (bottom). For in-depth technical details, please kindly refer to the respective manuscripts.

<!---
## Dataset
The raw file for [PPMI](https://www.ppmi-info.org/) dataset used for pre-training can be accessed in the `Data/` folder. The `.mat` file stores the adjacency connections of smapled brain networks from **three** different tractography algorithms (aka. views). The `.xlsx` file stores the atlas template of **Desikan-Killiany 84** parcellation system, and is further processed/dumped into a numpy `.arr` file for faster look-up.
-->
## Instructions
1. Run the main file will start the pre-training process, all parameters are defaulted to following:
- **backbone**, default = (GCN, GAT, GIN). The backbone encoder.
- **rdim**, type = `int`. The dimension the atlas mapping pre-preocessing reduce to.
- **filename**, type = `str`. The filename used to store the pre-trained parameters. Must be using the suffix `.pth`.
## Dependencies
The frameworks require the following dependencies to operate *(latest versions are recommended):*
```
torch~=1.10.2
numpy~=1.22.2
scikit-learn~=1.0.2
networkx~=2.6.2
scipy~=1.8.0
tqdm~=4.62.3
torch-geometric~=2.0.3
higher~=0.2.1
```
## Citation
Please cite our work if you find this repository helpful:
```bibtex
@inproceedings{yang2022data,
  title={Data-efficient brain connectome analysis via multi-task meta-learning},
  author={Yang, Yi and Zhu, Yanqiao and Cui, Hejie and Kan, Xuan and He, Lifang and Guo, Ying and Yang, Carl},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4743--4751},
  year={2022}
}
```
```bibtex
@inproceedings{yang2023ptgb,
  title={PTGB: Pre-Train Graph Neural Networks for Brain Network Analysis},
  author={Yang, Yi and Cui, Hejie and Yang, Carl},
  booktitle={Conference on Health, Inference, and Learning},
  pages={526--544},
  year={2023},
  organization={PMLR}
}
```

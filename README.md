# Pre-Training and Meta-Learning of Graph Neural Networks for Brain Network Analysis
<kbd> <img src="https://github.com/Owen-Yang-18/BrainNN-PreTrain/blob/main/Figures/meta.png"> </kbd>

<kbd> <img src="https://github.com/Owen-Yang-18/BrainNN-PreTrain/blob/main/Figures/pipeline.png"> </kbd>

This repository contains the implementation of the frameworks presented in the papers [PTGB: Pre-Train Graph Neural Networks for Brain Network Analysis (CHIL 2023 **Oral**)](https://arxiv.org/pdf/2305.14376.pdf) and [Data-Efficient Brain Connectome Analysis via Multi-Task Meta-Learning (SIGKDD 2022)](https://arxiv.org/pdf/2206.04486.pdf). The figures above present the visual overviews for multi-task meta-learning (top) and self-supervised pre-training (bottom). Please refer to the manuscripts for technical details.

## Instructions
1. Run the `main.py` file will start the pre-training process:
```bash
python main.py --dir=<path leading to the project directory>
```
The only parameter needing custom handling is the `--dir` argument, which stands for the local path leading to (not including) the project directory. Some other key parameters are defaulted as the following:
- **backbone**, default = GCN {GCN, GAT, GIN}. The backbone GNN encoder.
- **rdim**, default = 64, type = `int`. The dimension the atlas mapping pre-preocessing reduce to.
- **filename**, default = 'pretrained.pth', type = `str`. The filename used to store the pre-trained parameters. Must be using the suffix `.pth`.

2. How to prepare custom datasets:
- Please store your datasets into a `.npy` or `numpy` loadable file containing a python dictionary.
- Store the datasets for pre-training in the `data/source` folder, and for fine-tuning in the `data/target` folder.
- Delete the empty `source.txt` and `target.txt` files originally contained in the folders.
- For each dataset file, the dictionary is required to contain at least an `'adj'` attribute which stands for graph adjacencies, with the shape of $k \times n \times n$, where $k$ refers to number of samples, and $n$ stands for node/ROI quantity.
- The file can also recognize other attributes such as `'label'` for classifications, `'feat'` for node features, `'conn'` referring to the connectivity matrix of the ROIs' 3D coordinates.

3. Once pre-training is done, run the `'test.py'` file will begin the fine-tuning and testing process:
```bash
python test.py --dir=<path leading to the project directory>
```
Please make sure the parameters are set identically to those in the pre-training `main.py` setups.

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

import networkx as nx
import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
from networkx.convert_matrix import from_numpy_array
from scipy.io import loadmat
from torch import nn
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from tqdm.auto import tqdm

def distance_decoder(inputs, data):
    dist = torch.cdist(inputs, inputs)
    dist = torch.block_diag(*dist)
    return dist

def mapping_loss(data, bottleneck, x_hat, conn, mod):
    reconstruction_error = F.mse_loss(x_hat, data)
    locality_error = F.mse_loss(torch.matmul(conn, bottleneck), bottleneck)
    distance = distance_decoder(bottleneck, data)
    mod = torch.block_diag(*mod)
    modularity_error = (1.0 / (data.shape[0] * data.shape[1] * 2.0)) * torch.sum(mod * distance)
    return reconstruction_error, locality_error, modularity_error

class Autoencoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.proj1 = nn.Linear(inp, out, bias=False)
        torch.nn.init.normal_(self.proj1.weight)

    def forward(self, input):
        encoded_feats = F.dropout(self.proj1(input), 0.4)
        reconstructed_output = torch.matmul(encoded_feats, self.proj1.weight)
        return encoded_feats, reconstructed_output

def eigen_sort(transform, data):
    A = torch.matmul(data.flatten(0, 1).t(), data.flatten(0, 1))
    eigen_values = torch.mean(torch.div(torch.matmul(A, transform.t()), transform.t()), dim=0)
    sorted_eigen, indices = torch.sort(eigen_values, descending=True, stable=True)
    return F.normalize(transform[indices])

def train_ae(name, rdim):
    m = loadmat(f'Data/{name}.mat')

    if name == 'BP_DTI':
        data = torch.empty(97, 82, 82)
        mod = torch.empty_like(data)
        for i in range(97):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(m['X_normalize'][i, 0][:, :, 1])))
            data[i] = torch.Tensor(m['X_normalize'][i, 0][:, :, 1])
    elif name == 'HIV_DTI':
        data = F.relu(torch.Tensor(m['dti']).transpose(0, 2))
        mod = torch.empty_like(data)
        for i in range(data.shape[0]):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(data[i].numpy())))
    elif name == 'BP_FMRI' or name == 'HIV_FMRI':
        data = torch.Tensor(m['fmri']).transpose(0, 2)
        mod = torch.empty_like(data)
        for i in range(data.shape[0]):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(data[i].numpy())))
    elif name == 'PPMI_PICO':
        data = torch.empty(718, 84, 84)
        mod = torch.empty(718, 84, 84)
        for i in range(718):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(m['X'][i, 0][:, :, 0])))
            data[i] = torch.Tensor(m['X'][i, 0][:, :, 0])
    elif name == 'PPMI_HOUGH':
        data = torch.empty(718, 84, 84)
        mod = torch.empty(718, 84, 84)
        for i in range(718):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(m['X'][i, 0][:, :, 1])))
            data[i] = torch.Tensor(m['X'][i, 0][:, :, 1])
    elif name == 'PPMI_FSL':
        data = torch.empty(718, 84, 84)
        mod = torch.empty(718, 84, 84)
        for i in range(718):
            mod[i] = torch.Tensor(nx.modularity_matrix(from_numpy_array(m['X'][i, 0][:, :, 2])))
            data[i] = torch.Tensor(m['X'][i, 0][:, :, 2])

    density = torch.count_nonzero(data) / (data.shape[0] * data.shape[1] * data.shape[2])
    drop = 1.0 - density.item()
    AE = Autoencoder(data.shape[1], rdim)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.02)
    if data.shape[1] == 84:
        connectivity = torch.Tensor(np.load("PPMI_Conn.arr", allow_pickle=True))
    elif data.shape[1] == 82:
        connectivity = torch.Tensor(np.load("BP_Conn.arr", allow_pickle=True))
    elif data.shape[1] == 90:
        connectivity = torch.Tensor(np.load("HIV_Conn.arr", allow_pickle=True))

    id = np.arange(data.shape[0]).tolist()
    n = 20
    batch_idx = [id[i:i + n] for i in range(0, len(id), n)]

    with tqdm(total=100, desc="Progress") as pbar:
        for k in range(100):
            l = 0.0
            for idx in batch_idx:
                optimizer.zero_grad()
                bottleneck, x_hat = AE(data[idx])
                regularization_loss = 0.0
                for param in AE.parameters():
                    regularization_loss += torch.mean(torch.abs(param))
                rec, loc, modularity = mapping_loss(data[idx], bottleneck, x_hat, connectivity, mod[idx])
                if k < 50:
                    loss = rec
                else:
                    loss = 0.8 * rec + (0.1 * loc) + (0.1 - 0.1 * torch.tanh(modularity)) + 0.01 * regularization_loss
                loss.backward()
                optimizer.step()
                l += loss.item()
            pbar.set_postfix({"loss": l})
            pbar.update()

    for _, param1 in AE.named_parameters():
        transform = param1.detach()

    transform = eigen_sort(transform, data)
    feat = F.linear(data, transform)

    torch.save(transform, f'{name}_mapping.map')

    feat = F.dropout(torch.abs(feat), drop).detach()
    torch.save(feat, f'{name}.feature')
    torch.save(data, f'{name}.adj')
    torch.save(mod, f'{name}.mod')

class TransformedDataset(InMemoryDataset):
    def __init__(self, root, name, view, node_feature, label, transform=None, pre_transform=None):
        self.name = name
        self.view = view
        self.node_feature = node_feature
        self.label = label
        self.filename_postfix = str(pre_transform) if pre_transform is not None else None
        super(TransformedDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.mat'

    @property
    def processed_file_names(self):
        return f'data_{self.filename_postfix}.pt' if self.filename_postfix is not None else 'data.pt'

    def download(self):
        pass

    def process(self):
        m = loadmat(f'{self.name}.mat')
        a1 = torch.load(f'{self.name}.adj')
        x1 = torch.load(f'{self.name}.feature')

        y = torch.Tensor(m['label']).float()
        y[y == -1] = 0.0
        y[y == 1] = 1.0

        data_list = []
        for i in range(a1.shape[0]):
            edge_index_s, edge_attr_s = dense_to_sparse(a1[i])
            if(y[i] == self.label):
                data = Data(x=x1[i], edge_index=edge_index_s, edge_attr=edge_attr_s, y=y[i])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse
from tqdm.auto import tqdm


def rec_loss(data, x_hat):
    return F.mse_loss(x_hat, data)


def mapping_loss(data, bottleneck, x_hat, conn):
    reconstruction_error = F.mse_loss(x_hat, data)
    if conn is not None:
        locality_error = F.mse_loss(torch.matmul(conn, bottleneck), bottleneck)
        return 0.8 * reconstruction_error + 0.1 * locality_error
    else:
        return reconstruction_error


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


def train_ae(data, rdim, conn):
    density = torch.count_nonzero(data) / (data.shape[0] * data.shape[1] * data.shape[2])
    drop = 1.0 - density.item()
    AE = Autoencoder(data.shape[1], rdim)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.02)

    if conn is not None:
        connectivity = conn
    else:
        connectivity = None

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
                if k <= 50:
                    loss = rec_loss(data[idx], x_hat)
                else:
                    loss = mapping_loss(data[idx], bottleneck, x_hat, conn) + 0.01 * regularization_loss
                loss.backward()
                optimizer.step()
                l += loss.item()
            pbar.set_postfix({"loss": l})
            pbar.update()

    for _, param1 in AE.named_parameters():
        transform = param1.detach()

    transform = eigen_sort(transform, data)

    feat = F.linear(data, transform)
    feat = F.dropout(torch.abs(feat), drop).detach()
    return feat


class TransformedDataset(InMemoryDataset):
    def __init__(self, root, name, st, rdim, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.st = st
        self.rdim = rdim
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
        m = np.load(f'{self.root}/BrainNN-PreTrain/data/{self.st}/{self.name}', allow_pickle=True)
        if isinstance(m, np.ndarray):
            m = m.item()

        conn = None
        if 'conn' in m.keys():
            conn = torch.Tensor(m['conn'])

        adj = torch.Tensor(m['adj'])
        if 'feat' in m.keys():
            feat = torch.Tensor(m['feat'])
            new_feat = train_ae(feat, self.rdim, conn)
        else:
            new_feat = train_ae(adj, self.rdim, conn)

        if 'label' in m.keys():
            label = torch.Tensor(m['label']).float()
            label[label == -1] = 0.0
            label[label == 1] = 1.0
        else:
            label = torch.zeros(adj.shape[0])

        data_list = []
        for i in range(adj.shape[0]):
            edge_index_s, edge_attr_s = dense_to_sparse(adj[i])
            data = Data(x=new_feat[i], edge_index=edge_index_s, edge_attr=edge_attr_s, y=label[i])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

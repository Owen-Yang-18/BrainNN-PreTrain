import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import remove_self_loops

global device
device = torch.device("cuda:0")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, final_dim, activation, num_layers):
        super(GCN, self).__init__()
        self.activation = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        self.layers.append(GCNConv(hidden_dim, final_dim, cached=False))

    def forward(self, x, edge_index, edge_weight):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)

        return z

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, final_dim, activation, num_layers):
        super(GAT, self).__init__()
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.layers = torch.nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim, hidden_dim))
        self.layers.append(GATConv(hidden_dim, final_dim))

    def forward(self, x, edge_index):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index)
            bn = torch.nn.BatchNorm1d(z.shape[1])
            z = self.dropout(self.activation(bn(z)))

        return z

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GINConv(MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight):
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_weight))
        return out

    def message(self, x_j):
        return x_j

class GIN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GINConv(torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
        ),
            train_eps=False))
        for _ in range(num_layers - 1):
            self.layers.append(
                GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(hidden_dim),
                ),
                    train_eps=False))

    def forward(self, x, edge_index, edge_weight):
        z = x
        zs = []
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            zs.append(z)
        z = torch.cat(zs, dim=1)
        return z

class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, batch, x, edge_index, edge_weight):
        z = self.encoder(x, edge_index, edge_weight)
        return z

class Generator(torch.nn.Module):
    def __init__(self, num_layers):
        super(Generator, self).__init__()
        self.input_dim = 2 * num_layers
        self.device = torch.device("cuda:" + str('0')) if torch.cuda.is_available() else torch.device("cpu")
        self.regularizer = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, num_layers)
        )

    def forward(self, x):
        output = self.regularizer(x)
        output = torch.nn.Softmax(dim=0)(output)
        output /= (output.max() * 1200.0)
        return output
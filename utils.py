import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.utils import get_laplacian, to_dense_adj, to_dense_batch

from models import device

def gradient_gcn(regularizer, fnet, data, input, grad):
    gradient = []
    reg = []
    net = []

    ad1 = get_laplacian(edge_index=data.edge_index, edge_weight=data.edge_attr, normalization='sym')
    ad1 = to_dense_adj(edge_index=ad1[0], edge_attr=ad1[1], max_num_nodes=data.x.shape[0])[0, :, :]
    ad1.requires_grad_(False)
    ad1.to(device)

    for params in regularizer.parameters():
        reg.append(params.data.requires_grad_(True))

    alpha = torch.nn.ReLU()(torch.matmul(input, reg[0]) + reg[1])
    alpha = torch.matmul(alpha, reg[2].T) + reg[3]
    alpha = torch.nn.Softmax(dim=0)(alpha)
    alpha = alpha / (alpha.max() * 1200.0)

    for param in fnet.parameters():
        net.append(param.data.requires_grad_(False))

    for i in range(len(alpha)):
        net[i] = net[i] - alpha[i] * (grad[i].data + 0.0001 * net[i])

    for j in range(0, len(net), 2):
        if j == 0:
            out1 = torch.nn.ReLU()(torch.spmm(ad1, torch.matmul(data.x, net[j+1].T)) + net[j])
        else:
            out1 = torch.nn.ReLU()(torch.spmm(ad1, torch.matmul(out1, net[j+1].T)) + net[j])

    loss = sub_loss(out1, data)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()

    for k in range(len((reg))):
        gradient.append(reg[k].grad.data)

    return gradient

def gradient_gat(regularizer, fnet, data, input, grad):
    adj1 = to_dense_adj(data.edge_index, edge_attr=data.edge_attr, max_num_nodes=data.x.shape[0])[0, :, :].to(device)
    gradient = []
    reg = []
    net = []

    for params in regularizer.parameters():
        reg.append(params.data.requires_grad_(True))

    alpha = torch.nn.ReLU()(torch.matmul(input, reg[0]) + reg[1])
    alpha = torch.matmul(alpha, reg[2].T) + reg[3]
    alpha = torch.nn.Softmax(dim=0)(alpha)
    alpha = alpha / (alpha.max() * 1200.0)

    for param in fnet.parameters():
        net.append(param.data.requires_grad_(False))

    for i in range(len(alpha)):
        net[i] = net[i] - alpha[i] * (grad[i].data + 0.0001 * net[i])

    for j in range(0, len(net), 4):
        if j == 0:
            Wh = torch.mm(data.x, net[j + 3].T)
            Wh1 = torch.mm(Wh, net[j][0].T)
            Wh2 = torch.mm(Wh, net[j + 1][0].T)
            e = Wh1 + Wh2.T
            e = nn.LeakyReLU(0.3)(e)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj1 > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, 0.2)
            out1 = nn.ReLU()(torch.matmul(attention, Wh) + net[j + 2])
        else:
            Wh = torch.mm(out1, net[j + 3].T)
            Wh1 = torch.mm(Wh, net[j][0].T)
            Wh2 = torch.mm(Wh, net[j + 1][0].T)
            e = Wh1 + Wh2.T
            e = nn.LeakyReLU(0.3)(e)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj1 > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, 0.2)
            out1 = nn.ReLU()(torch.matmul(attention, Wh) + net[j + 2])

    loss = sub_loss(out1, data)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()

    for k in range(len((reg))):
        gradient.append(reg[k].grad.data)
    return gradient

def gradient_gin(regularizer, fnet, data, input, grad):
    gradient = []
    reg = []
    net = []
    z1 = []

    for params in regularizer.parameters():
        reg.append(params.data.requires_grad_(True))

    alpha = torch.nn.ReLU()(torch.matmul(input, reg[0]) + reg[1])
    alpha = torch.matmul(alpha, reg[2].T) + reg[3]
    alpha = torch.nn.Softmax(dim=0)(alpha)
    alpha = alpha / (alpha.max() * 1200.0)

    for param in fnet.parameters():
        net.append(param.data.requires_grad_(False))

    for i in range(len(alpha)):
        net[i] = net[i] - alpha[i] * (grad[i].data + 0.0001 * net[i])

    for j in range(0, len(net), 6):
        if j == 0:
            out1 = data.x + fnet.encoder.layers[int(j / 6)].propagate(data.edge_index, x=data.x,
                                                                      edge_attr=data.edge_attr)
            out1 = torch.nn.ReLU()(torch.matmul(out1, net[j].T) + net[j + 1])
            out1 = torch.nn.ReLU()(torch.matmul(out1, net[j + 2].T) + net[j + 3])
            out1 = (out1 * net[j + 4]) + net[j + 5]
            z1.append(out1)
        else:
            out1 = out1 + fnet.encoder.layers[int(j / 6)].propagate(data.edge_index, x=out1, edge_attr=data.edge_attr)
            out1 = torch.nn.ReLU()(torch.matmul(out1, net[j].T) + net[j + 1])
            out1 = torch.nn.ReLU()(torch.matmul(out1, net[j + 2].T) + net[j + 3])
            out1 = (out1 * net[j + 4]) + net[j + 5]
            z1.append(out1)

    out1 = torch.cat(z1, dim=1)

    loss = sub_loss(out1, data)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()

    for k in range(len((reg))):
        gradient.append(reg[k].grad.data)

    return gradient

def Adam(param, grad, alpha):
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    m_t = 0
    v_t = 0
    t = 0

    for _ in range(1):
        t += 1
        m_t = beta_1 * m_t + (1 - beta_1) * grad
        v_t = beta_2 * v_t + (1 - beta_2) * (torch.pow(grad, 2))
        m_cap = m_t / (1 - (beta_1 ** t))
        v_cap = v_t / (1 - (beta_2 ** t))
        param = param - (alpha * m_cap) / (torch.sqrt(v_cap) + epsilon)

    return param


def sub_loss(z, data):
    batch_z = to_dense_batch(z, data.batch)[0].to(device)
    batch_z = F.normalize(batch_z, dim=2)
    pos_mask = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
    num_nodes = batch_z.shape[1]
    num_graphs = batch_z.shape[0]
    pos_mask += torch.linalg.matrix_power(pos_mask, 2)
    pos_mask[pos_mask != 0.0] = 1.0
    neg_mask = 1. - pos_mask
    batch_idx = np.arange(0, batch_z.shape[0])

    loss = 0.0

    for j in range(batch_z.shape[0]):
        if j == 0:
            loss += intra_jsd(batch_z, batch_z, pos_mask, neg_mask)
        else:
            idx = np.roll(batch_idx, j).tolist()
            loss += inter_jsd(batch_z, batch_z, pos_mask, idx)

    loss /= num_nodes
    return loss


def intra_jsd(cand1, cand2, pos_mask, neg_mask):
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = torch.bmm(cand1, torch.transpose(cand2, 1, 2))

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos

    neg_sim = similarity * neg_mask
    E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos


def inter_jsd(cand1, cand2, pos_mask, idx):
    pos_mask = pos_mask + pos_mask[idx]
    pos_mask[pos_mask != 0.0] = 1.0

    neg_mask = 1.0 - pos_mask

    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = torch.bmm(cand1, torch.transpose(cand2[idx], 1, 2))

    E_pos = (np.log(2) - F.softplus(- similarity * pos_mask)).sum()
    E_pos /= num_pos

    neg_sim = similarity * neg_mask
    E_neg = (F.softplus(- neg_sim) + neg_sim - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos

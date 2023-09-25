import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import higher
import random
from models import device
from utils import gradient_gin, gradient_gcn, gradient_gat, sub_loss, Adam


def gradient(regularizer, fnet, d, rho, grad, backbone):
  if backbone == "GIN":
    return gradient_gin(regularizer, fnet, d, rho, grad)
  elif backbone == "GCN":
    return gradient_gcn(regularizer, fnet, d, rho, grad)
  elif backbone == "GAT":
    return gradient_gat(regularizer, fnet, d, rho, grad)

def train(encoder_model, regularizer, datasets, meta_opt, scheduler, epoch, backbone):
    encoder_model.train()
    regularizer.train()
    inner_opt = torch.optim.Adam(encoder_model.parameters(), lr=0.001, weight_decay=0.0001)
    qry_losses = []
    regularizer_gradient_container = []
    regularizer_final_gradient = []

    metasets = []
    updatesets = []

    for data in datasets:
        meta, upd = train_test_split(range(len(data)), train_size=0.5)
        metasets.append(DataLoader(Subset(data, random.sample(meta, 20)), batch_size=20))
        updatesets.append(DataLoader(Subset(data, random.sample(upd, 20)), batch_size=20))

    meta_opt.zero_grad()

    for i in range(len(metasets)):
      with higher.innerloop_ctx(
              encoder_model, inner_opt, copy_initial_weights=False
      ) as (fnet, diffopt):
          for dat in metasets[i]:
              dat = dat.to(device)
              z = fnet(dat.batch, dat.x, dat.edge_index, dat.edge_attr)
              spt_loss = sub_loss(z, dat)
              grad = torch.autograd.grad(spt_loss, fnet.parameters())
              rho = []
              for params in fnet.parameters():
                  rho.append(params.data.mean())
              for j in range(len(grad)):
                  rho.append(grad[j].mean())

              rho = torch.stack(rho)
              generated_alpha = regularizer(rho)

              for d in updatesets[i]:
                  d = d.to(device)
                  regularizer_gradient = gradient(regularizer, fnet, d, rho, grad, backbone)

              regularizer_gradient_container.append(regularizer_gradient)

              for g, param, alpha in zip(grad, fnet.parameters(), generated_alpha):
                  param.data = Adam(param.data, g, alpha)

          for data in updatesets[i]:
              data = data.to(device)
              z = fnet(data.batch, data.x, data.edge_index, data.edge_attr)
              qry_loss = sub_loss(z, data)
              qry_loss.backward()
              qry_losses.append(qry_loss.detach())

    meta_opt.step()
    scheduler.step()

    for j in range(len(regularizer_gradient_container[0])):
        a = torch.zeros(regularizer_gradient_container[0][j].shape).to(device)
        for l in range(len(regularizer_gradient_container)):
            a = a + regularizer_gradient_container[l][j]
        regularizer_final_gradient.append(a)

    for param1, m in zip(regularizer.parameters(), regularizer_final_gradient):
        param1.data = Adam(param1.data, m, alpha=0.001)

    return sum(qry_losses)

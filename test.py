import argparse
import numpy as np
import torch
import os

from torch import nn
from torch_geometric.data import DataLoader
from tqdm.auto import tqdm
from torch_geometric.nn import global_add_pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler

from dataset import TransformedDataset
from models import GIN, GCN, GAT, device


class DownstreamEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(DownstreamEncoder, self).__init__()
        self.encoder = encoder

    def forward(self, batch, x, edge_index, edge_weight):
        z = self.encoder(x, edge_index, edge_weight)
        g = global_add_pool(z, batch)
        return g

class DownstreamClassifier(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(DownstreamClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dim, 1))

    def forward(self, z):
        logits = self.classifier(z)
        return logits

def updater(finetune_encoder, classifier, trainloader, optimizer, scheduler):
    finetune_encoder.train()
    classifier.train()
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        g = finetune_encoder(data.batch, data.x, data.edge_index, data.edge_attr)
        z = classifier(g)
        loss_first = nn.BCEWithLogitsLoss()(z, data.y.view(-1,1))
        loss_first.backward()
        optimizer.step()
    scheduler.step()

    return loss_first.item()

def tester(finetune_encoder, classifier, testloader):
  roc = []
  acc = 0
  with torch.no_grad():
    for data in testloader:
        data = data.to(device)
        g = finetune_encoder(data.batch, data.x, data.edge_index, data.edge_attr)
        z = classifier(g)
        z = torch.nn.Sigmoid()(z)
        z = torch.transpose(z, 1, 0)
        z = z[0]
        z = z.detach().cpu().numpy()
        z = np.round(z)
        y = data.y.cpu().numpy()
        try:
            value = roc_auc_score(y, z, labels=2)
            roc.append(value * 100.0)
        except ValueError:
            roc.append(0.000)
        for i in range(len(y)):
            if(y[i] == z[i]):
                acc += 1
  return np.mean(roc) , acc / (len(testloader.sampler)) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='GCN', help='backbone encoder')
    parser.add_argument('--dir', type=str, default='/content/drive/MyDrive/', help='path to project directory')
    parser.add_argument('--rdim', type=int, default=64, help='reduced dimension for atlas mapping preprocessing')
    parser.add_argument('--hdim', type=int, default=32, help='hidden dimension for GNN encoder')
    parser.add_argument('--fdim', type=int, default=8, help='final hidden dimension for GNN encoder')
    parser.add_argument('--ddim', type=int, default=8, help='hidden dimension for downstream classifier')
    parser.add_argument('--filename', type=str, default='pretrained.pth', help='file name to store pre-trained weights')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 regularization for Adam optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience epoch for early stopping')
    parser.add_argument('--epoch', type=int, default=200, help='epochs for pre-training')
    parser.add_argument('--layers', type=int, default=4, help='number of layers')
    args = parser.parse_args()

    target = os.listdir(args.dir + 'BrainNN-PreTrain/data/target')
    target_data = []

    for name in target:
        target_data.append(TransformedDataset(root=args.dir, name=name, st='target', rdim=args.rdim))

    def test(backbone, rdim, file_name, ddim, dataset):
        ref_classifier = DownstreamClassifier(hidden_dim=ddim).to(device)
        kfold = KFold(n_splits=10, shuffle=True)
        accuracy, auroc = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
          if backbone == "GIN":
            gconv = GIN(input_dim=rdim, hidden_dim=8, activation='relu', num_layers=args.layers).to(device)
          elif backbone == "GCN":
            gconv = GCN(input_dim=rdim, hidden_dim=args.hdim, final_dim=args.fdim, activation='relu', num_layers=args.layers).to(device)
          elif backbone == "GAT":
            gconv = GAT(input_dim=rdim, hidden_dim=args.hdim, final_dim=args.fdim, activation='relu', num_layers=args.layers).to(device)
          else:
            AssertionError
          finetune_encoder = DownstreamEncoder(encoder=gconv).to(device)
          finetune_encoder.load_state_dict(torch.load(file_name))
          classifier = DownstreamClassifier(hidden_dim=ddim).to(device)
          classifier.load_state_dict(ref_classifier.state_dict())
          optimizer = torch.optim.Adam([
                  {'params': finetune_encoder.parameters()},
                  {'params': classifier.parameters()}
              ], lr=0.001, weight_decay=0.00001)
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=0.0001)

          train_subsampler = SubsetRandomSampler(train_ids)
          test_subsampler = SubsetRandomSampler(test_ids)

          trainloader = DataLoader(
              dataset,
              batch_size=20, sampler=train_subsampler)
          testloader = DataLoader(
              dataset,
              batch_size=20, sampler=test_subsampler)


          with tqdm(total=147, desc='(T)') as pbar:
              for i in range(147):
                  l = updater(finetune_encoder, classifier, trainloader, optimizer, scheduler)
                  _, Acc = tester(finetune_encoder, classifier, testloader)
                  pbar.set_postfix({"loss": l})
                  if Acc >= 95.0:
                    break
                  pbar.update()

          auc, acc = tester(finetune_encoder, classifier, testloader)
          accuracy.append(acc)
          auroc.append(auc)

    for dataset in target_data:
        test(backbone=args.backbone, rdim=args.rdim, file_name=args.filename, ddim=args.ddim, dataset=dataset)

if __name__ == '__main__':
    main()

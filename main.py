import argparse
import torch
import os
from tqdm.auto import tqdm
from dataset import TransformedDataset
from models import GIN, GAT, Encoder, Generator, GCN, device
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/content/drive/MyDrive/', help='path to project directory')
    parser.add_argument('--backbone', type=str, default='GCN', help='backbone encoder')
    parser.add_argument('--rdim', type=int, default=64, help='reduced dimension for atlas mapping preprocessing')
    parser.add_argument('--hdim', type=int, default=32, help='hidden dimension for GNN encoder')
    parser.add_argument('--fdim', type=int, default=8, help='final hidden dimension for GNN encoder')
    parser.add_argument('--filename', type=str, default='pretrained.pth', help='file name to store pre-trained weights')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 regularization for Adam optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience epoch for early stopping')
    parser.add_argument('--epoch', type=int, default=200, help='epochs for pre-training')
    parser.add_argument('--layers', type=int, default=4, help='number of layers')
    args = parser.parse_args()


    source = os.listdir(args.dir + 'BrainNN-PreTrain/data/source')

    source_data = []
    for name in source:
        source_data.append(TransformedDataset(root=args.dir, name=name, st='source', rdim=args.rdim))

    def driver(backbone, rdim, file_name):
        if backbone == "GIN":
            gconv = GIN(input_dim=rdim, hidden_dim=8, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv).to(device)
        elif backbone == "GCN":
            gconv = GCN(input_dim=rdim, hidden_dim=args.hdim, final_dim=args.fdim, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv).to(device)
        elif backbone == "GAT":
            gconv = GAT(input_dim=rdim, hidden_dim=args.hdim, final_dim=args.fdim, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv).to(device)
        else:
            AssertionError

        meta_opt = torch.optim.Adam(encoder_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=400, eta_min=0.0001)
        regularizer = Generator(num_layers=len(list(encoder_model.parameters()))).to(device)

        best = 1e9
        cnt_wait = 0
        patience = args.patience

        with tqdm(total=args.epoch, desc='Progress') as pbar:
            for i in range(args.epoch):
                loss = train(encoder_model, regularizer, source_data, meta_opt, scheduler, i, backbone)
                pbar.set_postfix({"loss": loss})
                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(encoder_model.state_dict(), file_name)
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!')
                    break
                pbar.update()

    driver(backbone=args.backbone, rdim=args.rdim, file_name=args.filename)

if __name__ == '__main__':
    main()

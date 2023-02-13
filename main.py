import argparse
import torch
from dataset import train_ae, TransformedDataset
from tqdm.auto import tqdm
from models import Encoder, Generator, device, GAT, GCN, GIN
from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='GCN', help='backbone encoder')
    parser.add_argument('--rdim', type=int, default=82, help='reduced dimension for atlas mapping preprocessing')
    parser.add_argument('--filename', type=str, default='GPTB GCN.pth', help='file name to store pre-trained weights')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='l2 regularization for Adam optimizer')
    parser.add_argument('--patience', type=int, default=20, help='patience epoch for early stopping')
    parser.add_argument('--epoch', type=int, default=200, help='epochs for pre-training')
    parser.add_argument('--layers', type=int, default=4, help='number of layers')
    args = parser.parse_args()

    train_ae("PPMI_PICO", args.rdim)
    train_ae("PPMI_HOUGH", args.rdim)
    train_ae("PPMI_FSL", args.rdim)

    ppmi_pico_pos = TransformedDataset(root='', name='PPMI_PICO', view='pico', node_feature='adjacency', label=1.0)
    ppmi_hough_pos = TransformedDataset(root='', name='PPMI_HOUGH', view='hough', node_feature='adjacency', label=1.0)
    ppmi_fsl_pos = TransformedDataset(root='', name='PPMI_FSL', view='fsl', node_feature='adjacency', label=1.0)

    ppmi_pico_neg = TransformedDataset(root='', name='PPMI_PICO', view='pico', node_feature='adjacency', label=0.0)
    ppmi_hough_neg = TransformedDataset(root='', name='PPMI_HOUGH', view='hough', node_feature='adjacency', label=0.0)
    ppmi_fsl_neg = TransformedDataset(root='', name='PPMI_FSL', view='fsl', node_feature='adjacency', label=0.0)

    def driver(backbone, rdim, file_name):
        pos = []
        neg = []

        neg.append(ppmi_pico_neg)
        neg.append(ppmi_hough_neg)
        neg.append(ppmi_fsl_neg)

        pos.append(ppmi_pico_pos)
        pos.append(ppmi_hough_pos)
        pos.append(ppmi_fsl_pos)

        if backbone == "GIN":
            gconv = GIN(input_dim=rdim, hidden_dim=8, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv, hidden_dim=32).to(device)
        elif backbone == "GCN":
            gconv = GCN(input_dim=rdim, hidden_dim=32, final_dim=8, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv, hidden_dim=8).to(device)
        elif backbone == "GAT":
            gconv = GAT(input_dim=rdim, hidden_dim=32, final_dim=8, activation='relu', num_layers=args.layers).to(device)
            encoder_model = Encoder(encoder=gconv, hidden_dim=8).to(device)
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
                loss = train(encoder_model, regularizer, pos, neg, meta_opt, scheduler, i, backbone)
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
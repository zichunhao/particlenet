import math
from pathlib import Path
from utils import (
    initialize_model, 
    initialize_dataloader, 
    initialize_optimizer,
    parse_args
)

import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

plt.switch_backend('agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(4)
torch.autograd.set_detect_anomaly(True)


def main(args):
    
    train_loader, test_loader = initialize_dataloader(args)
    C = initialize_model(args)
    C_optimizer = initialize_optimizer(args, C)

    if args.scheduler:
        steps_per_epoch = len(train_loader)
        if not args.load_model:
            cycle_last_epoch = 1
        else:
            cycle_last_epoch = (args.start_epoch * steps_per_epoch) - 1
        cycle_total_epochs = (2 * args.cycle_up_num_epochs) + args.cycle_cooldown_num_epochs

        C_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            C_optimizer,
            max_lr=args.cycle_max_lr,
            pct_start=(args.cycle_up_num_epochs / cycle_total_epochs),
            epochs=cycle_total_epochs,
            steps_per_epoch=steps_per_epoch,
            final_div_factor=args.cycle_final_lr / args.lr,
            anneal_strategy='linear',
            last_epoch=cycle_last_epoch
        )

    loss = torch.nn.BCELoss().to(device=args.device, dtype=args.dtype)

    train_losses = []
    valid_losses = []
    best_loss = math.inf
    
    # paths
    path_results = Path(args.path_results)
    path_loss = path_results / 'losses'
    path_roc = path_results / 'roc'
    path_weights = path_results / 'weights'
    
    path_results.mkdir(parents=True, exist_ok=True)
    path_loss.mkdir(parents=True, exist_ok=True)
    path_roc.mkdir(parents=True, exist_ok=True)
    path_weights.mkdir(parents=True, exist_ok=True)
    
    try:
        best_ep_info = torch.load(path_loss / 'best_loss.pt')
        best_loss = best_ep_info['best_loss']
    except FileNotFoundError:
        best_ep_info = {'best_loss': math.inf, 'best_epoch': 0}
    num_stale_epochs = 0

    def plot_losses(epoch, train_losses, valid_losses):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(train_losses)
        ax1.set_title('training')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(valid_losses)
        ax2.set_title('testing')

        plt.savefig(path_loss / f"loss-epoch_{str(epoch)}.pdf")
        plt.close()
        return 

    def save_model(epoch):
        torch.save(C.state_dict(), path_weights / f"weights-epoch_{str(epoch)}.pt")

    def train_C(data, y):
        C.train()
        C_optimizer.zero_grad()
        
        data = data.to(device=args.device, dtype=args.dtype)
        y = y.to(device=args.device, dtype=args.dtype)
        output = C(data)

        # nll_loss takes class labels as target, so one-hot encoding is not needed
        C_loss = loss(output, y.unsqueeze(-1))

        C_loss.backward()
        C_optimizer.step()

        return C_loss.item()

    def test(epoch):
        C.eval()
        valid_loss = 0
        y_outs = []
        logging.info("testing")
        dataset = test_loader.dataset
        
        with torch.no_grad():
            for batch_ndx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
                logging.debug(f"x[0]: {x[0]}, y: {y}")
                
                output = C(x.to(device=args.device, dtype=args.dtype))
                y = y.to(device=args.device, dtype=args.dtype)
                
                valid_loss += loss(output, y.unsqueeze(-1)).item()
                pred = output.squeeze()
                # logging.debug(f"pred: {pred}, output: {output}")

                y_outs.append(output.squeeze(-1))
            

        valid_loss /= len(test_loader)
        valid_losses.append(valid_loss)

        y_outs = torch.cat(y_outs)
        logging.debug(f"y_outs {y_outs}")
        logging.debug(f"y_true {dataset[:][1].numpy()}")

        acc = (y_outs.reshape(-1).cpu().round() == test_loader.dataset[:][1]).float().mean()
        
        y_outs = y_outs.cpu().numpy()
        # logging.info(dataset[:][1].numpy(), y_outs)
        fpr, tpr, _ = roc_curve(dataset[:][1].numpy(), y_outs)
        roc_auc = auc(fpr, tpr)
        if roc_auc < 0.5:
            # flip the sign of the output
            fpr, tpr, _ = roc_curve(1 - dataset[:][1].numpy(), y_outs)
            roc_auc = auc(fpr, tpr)
            
        torch.save({'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}, path_roc / f"roc-epoch_{epoch}-auc_{roc_auc:.4f}.pt")

        logging.info(
            f"{epoch=} Avg. loss: {valid_loss:.4f}, Accuracy: {acc:.4f}, AUC: {roc_auc:.4f}"
        )
    
        return valid_loss

    for i in range(args.start_epoch, args.num_epochs):
        logging.info(f"Epoch {i+1}")
        C_loss = 0
        
        logging.info("training")
        for batch_ndx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            C_loss += train_C(x.to(device), y.to(device))
            if args.scheduler:
                C_scheduler.step()

        train_losses.append(C_loss / len(train_loader))
        
        valid_loss = test(i)
        if valid_loss <= best_loss:
            best_loss = valid_loss
            best_ep_info = {
                'best_loss': best_loss,
                'best_epoch': i
            }
            torch.save(best_ep_info, path_results / 'best_loss.pt')
            num_stale_epochs = 0
        else:
            num_stale_epochs += 1
        logging.info(f"Best loss: {best_loss:.4f} (epoch {best_ep_info['best_epoch']}, {num_stale_epochs=})")

        if ((i + 1) % 1 == 0):
            save_model(i + 1)
            plot_losses(i + 1, train_losses, valid_losses)
            
        if (args.patience > 0) and (num_stale_epochs >= args.patience):
            logging.info(f"Early stopping after {i+1} epochs")
            break
    
    logging.info("Done!")
    logging.info(f"Best loss: {best_loss:.4f} (epoch {best_ep_info['best_epoch']})")


if __name__ == "__main__":
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parse_args()
    main(args)

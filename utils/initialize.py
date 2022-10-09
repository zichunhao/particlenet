
import argparse
import logging
from typing import Tuple
import torch

import sys
sys.path.insert(0, '../')
from model import ParticleNet
from .dataset import JetsClassifierDataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def initialize_model(args: argparse.ArgumentParser) -> ParticleNet:
    model = ParticleNet(
        num_hits=args.num_particles, 
        node_feat_size=args.node_feat_size,
        num_classes=2
    ).to(device=args.device, dtype=args.dtype)
    if args.load_model:
        model.load_state_dict(torch.load(args.path_model_weights))
    return model

def initialize_dataloader(
    args: argparse.ArgumentParser
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Initialize dataloaders for training and testing.

    :param args: Arguments from command line.
    :type args: argparse.ArgumentParser
    :return: (dataloader_train, dataloader_test)
    :rtype: Tuple[DataLoader, DataLoader]
    """ 
    # load data
    data_train_sig = torch.load(args.path_data_train_sig)
    data_train_bkg = torch.load(args.path_data_train_bkg)
    data_test_sig = torch.load(args.path_data_test_sig)
    data_test_bkg = torch.load(args.path_data_test_bkg)
    
    # initialize datasets
    dataset_train = JetsClassifierDataset(data_train_sig, data_train_bkg)
    dataset_test = JetsClassifierDataset(data_test_sig, data_test_bkg)
    
    # initialize dataloaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.test_batch_size,
        shuffle=True
    )
    return dataloader_train, dataloader_test

def initialize_optimizer(
    args: argparse.ArgumentParser, 
    model: ParticleNet
) -> Optimizer:
    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        logging.warning(f"{args.optimizer} is not a valid optimizer. Using AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

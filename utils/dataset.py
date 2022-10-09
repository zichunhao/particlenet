from typing import Tuple
import torch
from torch.utils.data import Dataset
import logging

class JetsClassifierDataset(Dataset):
    """Dataset for ParticleNet tagger. 
    Adapted from Raghav's code: https://github.com/rkansal47/mnist_graph_gan/blob/139a82282243a2b6cf201e9ee999a0a9a03e7b32/jets/jets_dataset.py#L99.
    """    
    def __init__(self, sig: torch.Tensor, bkg: torch.Tensor):
        """Dataset for ParticleNet tagger. 

        :param sig: Signal jets, assigned label 1.
        :type sig: torch.Tensor
        :param bkg: Background jets, assigned label 0.
        :type bkg: torch.Tensor
        """
        logging.info(f"{sig.shape=}")
        logging.info(f"{bkg.shape=}")
        # data
        self.X = torch.cat((sig, bkg), dim=0)
        # labels
        self.Y = torch.cat((torch.ones(len(sig)), torch.zeros(len(bkg))), dim=0)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
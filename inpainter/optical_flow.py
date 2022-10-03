from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from RAFT import RAFT


def initialize_RAFT(args, device):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model))

    model = model.module
    model.to(device)
    model.eval()

    return model


class BaseOpticalFLowEstimator(ABC):

    @abstractmethod
    def estimate(self, frames) -> Tuple[List, List]:
        pass


class RaftOpticalFlowEstimator(BaseOpticalFLowEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.model = initialize_RAFT()


    def estimate(self, frames) -> Tuple[List, List]:
        for frame in frames:
            

    
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from enum import Enum
from tqdm import tqdm

import torch
from torch import nn
import numpy as np

from RAFT import RAFT
from LAFC.models.lafc import Model as LAFC
import tool.utils.region_fill as rf


class FlowDirection(Enum):
    FORWARD = 0
    BACKWARD = 1


class BaseOpticalFlowEstimator(ABC):

    @abstractmethod
    def _predict(self, image_0: np.ndarray , image_1: np.ndarray, *args, **kwargs) -> np.ndarray:
        pass
        
    def estimate_flows(self, frames: np.ndarray , mode: FlowDirection) -> Tuple[List, List]:
        result = []
        with torch.no_grad():
            for idx in tqdm(range(len(frames)-1)):
                image_0, image_1 = (frames[idx, None], frames[idx + 1, None]) if mode is FlowDirection.FORWARD \
                    else (frames[idx + 1, None], frames[idx, None])
                
                flow = self._predict(image_0, image_1)
                result.append(flow)

        result = np.asarray(result)
        return result

    def estimate_fb_flows(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print('Optical flows estimation. Forward and backward directaions')
        forward_flows = self.estimate_flows(frames, FlowDirection.FORWARD)
        backward_flows = self.estimate_flows(frames, FlowDirection.BACKWARD)
        return forward_flows, backward_flows


class RAFTOpticalFlowEstimator(BaseOpticalFlowEstimator):
    def __init__(self, model: Union[str, nn.Module] = 'LAFC/flowCheckPoint/raft-things.pth', device='cuda:0') -> None:
        super().__init__()
        
        self.device = device
        if isinstance(model, str):
            self.model = self.initialize_RAFT(model, device)
        else:
            self.model = model.to(device)

    @staticmethod
    def initialize_RAFT(model_path: str, device='cuda:0'):
        
        model = RAFT()
        model = torch.nn.DataParallel(model)
        tmp = torch.load(model_path)
        model.load_state_dict(tmp)

        model = model.module
        model.to(device)
        model.eval()
        return model

    def _predict(self, image_0: np.ndarray, image_1: np.ndarray, *args, **kwargs) -> np.ndarray:
        
        image_0, image_1 = np.moveaxis(image_0, -1, 1), np.moveaxis(image_1, -1, 1)
        image_0, image_1 = torch.tensor(image_0), torch.tensor(image_1)
        image_0, image_1 = image_0.to(self.device), image_1.to(self.device)
        
        _, flow = self.model(image_0, image_1, iters=20, test_mode=True)
        del image_0, image_1
        torch.cuda.empty_cache()
        
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow


class BaseOpticalFlowCompleter(ABC):

    @abstractmethod
    def _complete(self, flows: np.ndarray, masks: np.ndarray, mode: FlowDirection) -> np.ndarray:
        pass
    
    def complete(self, flows: np.ndarray, masks: np.ndarray, mode: FlowDirection) -> np.ndarray:
        flows, masks = self.preprocess(flows, masks)
        flows = self._complete(flows, masks, mode)
        return flows
    
    def preprocess(self, flows: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        masks = masks/255
        return flows, masks

    def fb_complete(self, forward_flows: np.ndarray, backward_flows: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print('Optical flows completion. Forward and backward directaions')
        forward_flows = self.complete(forward_flows, masks, FlowDirection.FORWARD)
        backward_flows = self.complete(backward_flows, masks, FlowDirection.BACKWARD)
        return forward_flows, backward_flows


class LAFCOpticalFlowCompleter(BaseOpticalFlowCompleter):
    def __init__(self, model: Union[str, nn.Module] = 'LAFC/checkpoint/lafc.pth.tar', device: str = 'cuda:0') -> None:
        super().__init__()
        if isinstance(model, str):
            self.model = self.initialize_LAFC(model, device)
        else:
            self.model = model.to(device)
        self.device = device

    @staticmethod
    def initialize_LAFC(pth: str, device='cuda:0') -> nn.Module:
        model = LAFC()
        state = torch.load(pth, map_location=lambda storage, loc: storage.cuda(device))
        model.load_state_dict(state['model_state_dict'])
        model = model.to(device)
        return model

    @staticmethod
    def diffusion(flows: np.ndarray, masks: np.ndarray) -> List[np.ndarray]:
        flows_filled = []
        for i in range(len(flows)):
            flow, mask = flows[i], masks[i]
            flow_filled = np.zeros(flow.shape)
            flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0], mask[:, :, 0])
            flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1], mask[:, :, 0])
            flows_filled.append(flow_filled)
        flows_filled = np.asarray(flows_filled)
        return flows_filled

    @staticmethod
    def _2tensor(array: np.ndarray) -> torch.tensor:
        array = np.asarray(array)
        array = torch.from_numpy(np.transpose(array, (3, 0, 1, 2))).unsqueeze(0).float()  # [1, c, t, h, w]
        return array

    @staticmethod
    def generate_idxs(pivot, interval, frames, t) -> List[int]:
        singleSide = frames // 2
        results = []
        for i in range(-singleSide, singleSide + 1):
            index = pivot + interval * i
            if index < 0:
                index = abs(index)
            if index > t - 1:
                index = 2 * (t - 1) - index
            results.append(index)

        return results

    def _complete(self, flows: np.ndarray, masks: np.ndarray, mode: FlowDirection) -> np.ndarray:
        # flow_masks  [N, H, W]
        # flows [N, H, W, 2]
    
        num_flows = 3
        flow_interval = 3

        masks = masks[:, :, :, np.newaxis]
        masks = masks[:-1] if mode is FlowDirection.FORWARD else masks[1:]
        
        diffused_flows = self.diffusion(flows, masks)
        flows, masks, diffused_flows = self._2tensor(flows), self._2tensor(masks), self._2tensor(diffused_flows)
        
        t = diffused_flows.shape[2]
        pivot = num_flows // 2
        result = []

        for i in tqdm(range(t)):
            indices = self.generate_idxs(i, flow_interval, num_flows, t)
            cand_flows = flows[:, :, indices]
            cand_masks = masks[:, :, indices]
            inputs = diffused_flows[:, :, indices]
            pivot_mask = cand_masks[:, :, pivot]
            pivot_flow = cand_flows[:, :, pivot]
            with torch.no_grad():
                inputs, cand_masks = inputs.to(self.device), cand_masks.to(self.device)
                output_flow = self.model(inputs, cand_masks)[0].cpu()
                del inputs, cand_masks
                torch.cuda.empty_cache
                
            comp = output_flow * pivot_mask + pivot_flow * (1 - pivot_mask)
            result.append(comp[0].numpy())
                
        result = np.asarray(result)
        result = np.moveaxis(result, 1, -1)
        return result

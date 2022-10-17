from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from enum import Enum

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

        from tqdm import tqdm
        with torch.no_grad():

            for idx in tqdm(range(len(frames)-1)):
                
                if mode is FlowDirection.FORWARD:
                    image_0 = frames[idx, None]
                    image_1 = frames[idx + 1, None]

                else:
                    image_0 = frames[idx + 1, None]
                    image_1 = frames[idx, None]
                
                flow = self._predict(image_0, image_1)
                result.append(flow)

        result = np.asarray(result)
        return result

    def estimate_fb_flows(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def complete(self, flows: np.ndarray, masks: np.ndarray, mode: FlowDirection) -> np.ndarray:
        pass

    def fb_complete(self, forward_flows: np.ndarray, backward_flows: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def tensor(array: np.ndarray) -> torch.tensor:
        array = np.asarray(array)
        array = torch.from_numpy(np.transpose(array, (3, 0, 1, 2))).unsqueeze(0).float()  # [1, c, t, h, w]
        return array.to('cuda:0')

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

    def complete(self, flows: np.ndarray, masks: np.ndarray, mode: FlowDirection) -> np.ndarray:
        # flow_masks  [N, H, W]
        # flows [N, H, W, 2]
    
        num_flows = 3
        flow_interval = 3

        masks = masks[:, :, :, np.newaxis]
        masks = masks[:-1] if mode is FlowDirection.FORWARD else masks[1:]
        
        diffused_flows = self.diffusion(flows, masks)
        print('diffused_flows', diffused_flows.shape)
        
        print('diffused')
        import matplotlib.pyplot as plt
        plt.imshow(diffused_flows[10][:,:,0])
        plt.show()
        
        flows, masks, diffused_flows = self.tensor(flows), self.tensor(masks), self.tensor(diffused_flows)

        t = diffused_flows.shape[2]
        filled_flows = [None] * t
        pivot = num_flows // 2

        from tqdm import tqdm
        for i in tqdm(range(t)):
            indices = self.generate_idxs(i, flow_interval, num_flows, t)
            cand_flows = flows[:, :, indices]
            cand_masks = masks[:, :, indices]
            inputs = diffused_flows[:, :, indices]
            pivot_mask = cand_masks[:, :, pivot]
            pivot_flow = cand_flows[:, :, pivot]
            with torch.no_grad():
                output_flow = self.model(inputs, cand_masks)
            if isinstance(output_flow, tuple) or isinstance(output_flow, list):
                output_flow = output_flow[0]
                
            output_flow = output_flow.cpu()
            comp = output_flow.cpu() * pivot_mask.cpu() + pivot_flow.cpu() * (1 - pivot_mask.cpu())
            if filled_flows[i] is None:
                filled_flows[i] = comp
                
        filled_flows = [f[0].cpu().numpy() for f in filled_flows]
        filled_flows = np.asarray(filled_flows)
        filled_flows = np.moveaxis(filled_flows, 1, -1)
        
        print('filled_flows', filled_flows.shape)
        return filled_flows

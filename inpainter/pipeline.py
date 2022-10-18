from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, List

import numpy as np

from inpainter.inpainter import (
    BaseVideoInpainter,
)
from inpainter.optical_flow import (
    BaseOpticalFlowEstimator,
    BaseOpticalFlowCompleter,
)
from inpainter.utils import resize


class BaseVideoInpaintingPipeline(ABC):

    def preprocess(self, frames: List[np.ndarray], masks: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        frames = np.asarray(frames)
        masks = np.asarray(masks)
        return frames, masks

    def postprocess(self, frames: Union[List, np.ndarray], masks: Union[List, np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        frames = np.asarray(frames)
        masks = np.asarray(masks)
        return frames, masks

    @abstractmethod
    def _inpaint(self, frames: np.ndarray, masks: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def run(self, frames: np.ndarray, masks: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        initial_height, initial_width = frames[0].shape[:2]
        height, width = (initial_height, initial_width) if self.resolution is None else self.resolution
                
        resized_frames = resize(frames, height, width)
        resized_masks = resize(masks, height, width)
        
        inpainted_frames, inpainted_masks = self._inpaint(resized_frames, resized_masks, **kwargs)

        inpainted_frames = resize(inpainted_frames, initial_height, initial_width)
        inpainted_masks = resize(inpainted_masks, initial_height, initial_width)

        result_frames = []
        for frame, inp_frame, mask in zip(frames, inpainted_frames, masks):
            mask = mask > 0
            frame[mask] = inp_frame[mask]
            result_frames.append(frame)

        return result_frames, inpainted_masks


class VideoInpaintingPipeline(BaseVideoInpaintingPipeline):
    def __init__(
                self,
                propagation_inpainter: Union[str, BaseVideoInpainter],
                hallucination_inpainter: Union[str, BaseVideoInpainter],
                optical_flow_estimator: Union[str, BaseOpticalFlowEstimator],
                optical_flow_completer: Union[str, BaseOpticalFlowCompleter],

                resolution: Optional[Tuple[int, int]] = None,
                flow_propagation_steps: int = 1,
            ) -> None:
        super().__init__()

        self.resolution = resolution
        self.flow_propagation_steps = flow_propagation_steps

        self.propagation_inpainter = propagation_inpainter
        self.hallucination_inpainter = hallucination_inpainter
        self.optical_flow_estimator = optical_flow_estimator
        self.optical_flow_completer = optical_flow_completer

    def inpainted_optical_flow(self, frames: np.ndarray, masks: np.ndarray) -> Tuple[List, List]:
        forward_flow, backward_flow = self.optical_flow_estimator.estimate_fb_flows(frames)
        forward_flow, backward_flow = self.optical_flow_completer.fb_complete(frames, masks)
        return forward_flow, backward_flow

    def _inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(self.flow_propagation_steps):
            print(f'Gradient propagation stage {_+1}/{self.flow_propagation_steps}')

            forward_flow, backward_flow = self.optical_flow_estimator.estimate_fb_flows(frames)            
            forward_flow, backward_flow = self.optical_flow_completer.fb_complete(forward_flow, backward_flow, masks)
            frames, masks = self.propagation_inpainter.inpaint(frames, masks, forward_flow, backward_flow)

        print('Hallucination stage')
        frames, masks = self.hallucination_inpainter(frames, masks)
        return frames, masks

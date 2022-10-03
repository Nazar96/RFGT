from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Any, List

from inpainter import GradientPropagationInpainter, TransformerVideoInpainter


class BaseVideoInpaintingPipeline(ABC):

    @abstractmethod
    def init_models(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def preprocess(self, **kwrags) -> Any:
        pass

    @abstractmethod
    def postprocess(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


class VideoInpaintingPipeline(BaseVideoInpaintingPipeline):
    def __init__(
                self,
                optical_flow_model: str,
                inpainter: str,
                config: str,
                refine: bool = False,
            ) -> None:
        super().__init__()
        self.flow_inpainter = GradientPropagationInpainter()
        self.hallucination_inpainter = TransformerVideoInpainter()
        
        self.optical_flow_estimator = None
        self.optical_flow_restorer = None

    def complete_optical_flow(self, frames, masks) -> Tuple[List, List]:
        forward_flow, backward_flow = self.optical_flow_estimator(frames)
        forward_flow = self.optical_flow_restorer(forward_flow, masks)
        backward_flow = self.optical_flow_restorer(backward_flow, masks)
        return forward_flow, backward_flow

    def run(self, **kwargs) -> Any:
        frames, masks = self.load_data(**kwargs)
        frames, masks = self.preprocess(frames, masks, **kwargs)

        forward_flow, backward_flow = self.complete_optical_flow(forward_flow, backward_flow, masks)

        frames, masks = self.flow_inpainter(frames, masks, (forward_flow, backward_flow))
        frames, masks = self.hallucination_inpainter(frames, masks, forward_flow)
    
        frames, masks = self.postprocess(frames, masks)
        return frames, masks

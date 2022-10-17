from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np
import cv2

from tool.get_flowNN_gradient import get_flowNN_gradient
from tool.utils.Poisson_blend_img import Poisson_blend_img


class BaseVideoInpainter(ABC):
    """
    Base class for restoring the selected region in an video sequence. 
    """

    def preprocess_input(self, frames, masks, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return frames, masks

    def postprocess_output(self, frames, masks, *args, **kwargs):
        return frames, masks

    @abstractmethod
    def _inpaint(self, frames: np.ndarray, masks: np.ndarray, *args, **kwargs) -> Tuple[List[Any], List[Any]]:
        pass

    def inpaint(self, frames: np.ndarray, masks: np.ndarray, *args, **kwargs) -> Tuple[List[Any], List[Any]]:
        frames, masks = self.preprocess_input(frames, masks)
        frames, masks = self._inpaint(frames, masks)
        frames, masks = self.postprocess_output(frames, masks)
        return frames, masks

    def __call__(self, frames, masks, **kwargs) -> Any:
        return self.inpaint(frames, masks, **kwargs)


class NeighbourVideoInpainter(BaseVideoInpainter):
    """
    Restores the selected region in an video sequence using the region neighborhood. 
    """

    def __init__(self) -> None:
        super().__init__()

    def _inpaint(self, frames: np.ndarray, masks: np.ndarray, *args, **kwargs) -> Tuple[List[Any], List[Any]]:
        inpainted_frames = []
        inpainted_masks = []
        for idx in range(len(frames)):
            frame, mask = frames[idx], masks[idx]
            frame[mask!=0] = 0
            
            inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            inpainted_mask = np.zeros_like(mask)
            
            inpainted_frames.append(inpainted_frame)
            inpainted_masks.append(inpainted_mask)

        inpainted_frames, inpainted_masks = np.asarray(inpainted_frames), np.asarray(inpainted_masks)
        return inpainted_frames, inpainted_masks


class GradientPropagationVideoInpainter(BaseVideoInpainter):
    """
    Restores the selected region in an video sequence using content gradient propagation. 
    """

    def __init__(self) -> None:
        super().__init__()

    def preprocess_input(
                self, 
                frames: np.ndarray, 
                masks: np.ndarray, 
                forward_flows: np.ndarray, 
                backward_flows: np.ndarray,  
                *args, 
                **kwargs,
            ) -> Tuple[np.ndarray, np.ndarray]:

        frames = np.asarray(frames)
        masks = np.asarray(masks)

        frames = frames / 255
        masks = masks > 0

        return frames, masks, forward_flows, backward_flows

    def postprocess_output(self, frames: np.ndarray, masks: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        
        frames = np.asarray(frames).clip(0,1)
        masks = np.asarray(masks).clip(0,1)

        frames = (frames*255).astype(np.uint8)
        masks = (masks*255).astype(np.uint8)

        return frames, masks

    @staticmethod
    def gradient_mask(mask: np.ndarray) -> np.ndarray:
        return np.logical_or.reduce((
            mask,
            np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
            np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1),
            ))

    def prepare_gradients(self, frames:np.ndarray, masks:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare image gradients for further content propagation

        Args:
            frames (np.ndarray): sequence of frames
            masks (np.ndarray): sequence of masks

        Returns:
            Tuple[np.ndarray, np.ndarray]: gradients for content propagation
        """
        number_of_frames, height, width = frames.shape[:3]
        gradient_x = np.empty(((height, width, 3, 0)), dtype=np.float32)
        gradient_y = np.empty(((height, width, 3, 0)), dtype=np.float32)
        for idx in range(number_of_frames):
            frame, mask = frames[idx], self.gradient_mask(masks[idx])

            gradient_x_ = np.concatenate(
                (np.diff(frame, axis=1), np.zeros((height, 1, 3),
                dtype=np.float32)),
                axis=1,
                )
            gradient_y_ = np.concatenate(
                (np.diff(frame, axis=0), np.zeros((1, width, 3), 
                dtype=np.float32)), 
                axis=0
                )

            gradient_x_ = np.expand_dims(gradient_x_, -1)
            gradient_y_ = np.expand_dims(gradient_y_, -1)

            gradient_x = np.concatenate((gradient_x, gradient_x_), axis=-1)
            gradient_y = np.concatenate((gradient_y, gradient_y_), axis=-1)

            gradient_x[mask>0, :, idx] = 0
            gradient_y[mask>0, :, idx] = 0

        return gradient_x, gradient_y

    @staticmethod
    def poisson_blending(
                frames: np.ndarray,
                masks : np.ndarray,
                gradients_x: np.ndarray, 
                gradients_y: np.ndarray, 
                masks_gradient: np.ndarray,
            ) -> Tuple[List[Any], List[Any]]:
        """
        Propagate gradients using poisson blending 

        Args:
            frames (_type_):  sequence of frames
            masks (_type_):  sequence of frames
            gradients_x (_type_):  video sequence gradients along x 
            gradients_y (_type_):  video sequence gradients along y
            masks_gradient (_type_):  mask sequence gradients

        Returns:
            Tuple[List[Any], List[Any]]: propagated contnent frames nad masks
        """

        blended_frames = []
        blended_masks = []
        
        from tqdm import tqdm
        for idx in tqdm(range(len(frames))):
            frame, mask = frames[idx], masks[idx]
            gradient_x, gradient_y = gradients_x[:,:,:,idx], gradients_y[:,:,:,idx]
            mask_gradient = masks_gradient[:,:,idx] if masks_gradient is not None else None
            if mask.sum() > 0:
                frame, mask = Poisson_blend_img(
                    frame,
                    gradient_x,
                    gradient_y,
                    mask, 
                    mask_gradient,
                    )
                blended_frames.append(frame)
                blended_masks.append(mask)
        
        blended_frames = np.array(blended_frames)
        blended_masks = np.array(blended_masks)
        return blended_frames, blended_masks

    def _inpaint(
                self, 
                frames: np.ndarray, 
                masks: np.ndarray, 
                forward_flows: np.ndarray, 
                backward_flows: np.ndarray, 
                *args, 
                **kwargs
            ) -> Tuple[List[Any], List[Any]]:
        gx, gy = self.prepare_gradients(frames, masks)
        gx, gy, gm = get_flowNN_gradient(gx, gy, masks, forward_flows, backward_flows)
        frames, masks = self.poisson_blending(frames, masks, gx, gy, gm)
        return frames, masks

    def inpaint(
                self, 
                frames: np.ndarray, 
                masks: np.ndarray, 
                forward_flows: np.ndarray, 
                backward_flows: np.ndarray,
                *args,
                **kwargs
            ) -> Tuple[np.ndarray, np.ndarray]:
        frames, masks, forward_flows, backward_flows = self.preprocess_input(frames, masks, forward_flows, backward_flows)
        frames, masks = self._inpaint(frames, masks, forward_flows, backward_flows)
        frames, masks = self.postprocess_output(frames, masks) 
        return frames, masks

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any
import numpy as np
import cv2

from tool.get_flowNN_gradient import get_flowNN_gradient
from tool.utils.Poisson_blend_img import Poisson_blend_img


class BaseVideoInpainter(ABC):

    @abstractmethod
    def inpaint(self, frames: np.ndarray, masks: np.ndarray, flows: Optional[np.ndarray], **kwargs) -> Tuple[List[Any], List[Any]]:
        pass

    def __call__(self, frames, masks, **kwargs) -> Any:
        return self.inpaint(frames, masks, **kwargs)


class NeighbourVideoInpainter(BaseVideoInpainter):
    def __init__(self) -> None:
        super().__init__()

    def inpaint(self, frames: np.ndarray, masks: np.ndarray, flows: Optional[np.ndarray]=None, **kwargs) -> Tuple[List[Any], List[Any]]:
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
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def gradient_mask(mask: "mask") -> "mask":
        return np.logical_or.reduce((
            mask,
            np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
            np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1),
            ))

    def prepare_gradients(self, frames, masks) -> "gradients":
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
    def poisson_blending(frames, masks, gradients_x, gradients_y, masks_gradient) -> Tuple[List[Any], List[Any]]:
        blended_frames = []
        blended_masks = []
        for idx in range(len(frames)):
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
                blended_masks.append(masks)
        return blended_frames, blended_masks

    def inpaint(self, frames: np.ndarray, masks: np.ndarray, flows: Optional[Tuple], **kwargs) -> Tuple[List[Any], List[Any]]:
        frames, _ = NeighbourVideoInpainter().inpaint(frames, masks)
        gx, gy = self.prepare_gradients(frames, masks)
        gx, gy, gm = get_flowNN_gradient(gx, gy, masks>0, flows, flows)

        print()
        print(gx.shape)
        print(gy.shape)
        print(gm.shape)

        frames, masks = self.poisson_blending(frames, masks, gx, gy, gm)
        return frames, masks


class TransformerVideoInpainter(BaseVideoInpainter):
    pass

# from inpainter.inpainter import GradientPropagationVideoInpainter
# import numpy as np

# def fake_data(h=240, w=432, n=120):
#     images = np.random.uniform(size=(n, h, w, 3))*256
#     images = images.astype(np.uint8)

#     size = 30

#     masks = np.zeros((n, h, w), dtype=np.uint8)
#     masks[:, h//2-size : h//2+size, w//2-size : w//2+size] = 255

#     flows = np.random.uniform(size=(n, h, w, 2)).astype(np.float32)
#     return images, masks, flows

# a = GradientPropagationVideoInpainter()
# images, masks, flows = fake_data(240, 432, 60)
# a.inpaint(images, masks, flows)

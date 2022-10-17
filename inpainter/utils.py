from typing import List, Tuple
import os

import numpy as np
import cv2


def load_file_names(path: str) -> List[str]:
    files = os.listdir(path)
    files = sorted(files, key=lambda x: int(x.split('.')[-2]))
    return files


def load_frames(path: str, files: List[str]) -> List[np.ndarray]:
    frames = []
    for f in files:
        frame = cv2.imread(os.path.join(path, f))
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def load_masks(path: str, files: List[str]) -> List[np.ndarray]:
    masks = []
    for f in files:
        mask = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks


def load_data(frames_path: str, masks_path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    names = load_file_names(frames_path)
    return load_frames(frames_path, names), load_masks(masks_path, names), names


def save_frames(path: str, files: List[str], frames: List[np.ndarray]) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    for file, frame in zip(files, frames):
        file_path = os.path.join(path, file)
        cv2.imwrite(file_path, frame)


def resize(images: List[np.ndarray], h: int, w: int) -> List[np.ndarray]:
    return np.asarray([cv2.resize(img, (w, h)) for img in images])

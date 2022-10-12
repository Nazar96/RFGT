import argparse
from typing import Any

from inpainter.inpainter import GradientPropagationVideoInpainter, NeighbourVideoInpainter
from inpainter.optical_flow import RAFTOpticalFlowEstimator, LAFCOpticalFlowCompleter
from inpainter.pipeline import VideoInpaintingPipeline
from inpainter.utils import (
    load_data,
    save_frames,
)


def main(args: Any) -> None:

    propagation_inpainter = GradientPropagationVideoInpainter()
    hallucination_inpainter = NeighbourVideoInpainter()
    optical_flow_estimator = RAFTOpticalFlowEstimator(args.raft_path)
    optical_flow_completer = LAFCOpticalFlowCompleter(args.lafc_path)

    pipeline = VideoInpaintingPipeline(
        propagation_inpainter=propagation_inpainter,
        hallucination_inpainter=hallucination_inpainter,
        optical_flow_estimator=optical_flow_estimator,
        optical_flow_completer=optical_flow_completer,

        resolution=(args.height, args.width),
        flow_propagation_steps=args.n_propagation,
    )

    frames, masks, names = load_data(args.frames_path, args.masks_path)
    inpainted_frames, _ = pipeline.run(frames, masks)
    save_frames(args.output_path, names, inpainted_frames)

    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Video Inpainting Pipeline')

    parser.add_argument('--frames_path', type=str, help='Frames folder path')
    parser.add_argument('--masks_path', type=str, help='Masks folder path')
    parser.add_argument('--output_path', type=str, help='Video inpainting results path')

    parser.add_argument('--raft_path', type=str, help='RAFT model path', default='LAFC/flowCheckPoint/raft-things.pth')
    parser.add_argument('--laft_path', type=str, help='LAFT model path', default='LAFC/checkpoint/lafc.tar')

    parser.add_argument('--height', type=int, help='Internal inpainting resolution height')
    parser.add_argument('--width', type=int, help='Internal inpainting resolution width')

    parser.add_argument('--n_propagation', type=int, help='Number of gradient propagation inpainting iterations', default=1)

    parser.add_argument('--gpu', type=bool, default=True)

    args = parser.parse_args()
    main(args)

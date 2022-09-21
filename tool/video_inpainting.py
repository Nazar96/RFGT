import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'FGT')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', 'LAFC')))

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import cv2
import glob
import copy
import numpy as np
import torch
import imageio
from PIL import Image
import scipy.ndimage
from skimage.feature import canny
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from RAFT import utils
from RAFT import RAFT
import yaml
from importlib import import_module

import utils.region_fill as rf
from utils.Poisson_blend_img import Poisson_blend_img
from get_flowNN_gradient import get_flowNN_gradient
from torchvision.transforms import ToTensor
import cvbase

import  time


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def diffusion(flows, masks):
    flows_filled = []
    for i in range(flows.shape[0]):
        flow, mask = flows[i], masks[i]
        flow_filled = np.zeros(flow.shape)
        flow_filled[:, :, 0] = rf.regionfill(flow[:, :, 0], mask[:, :, 0])
        flow_filled[:, :, 1] = rf.regionfill(flow[:, :, 1], mask[:, :, 0])
        flows_filled.append(flow_filled)
    return flows_filled


def np2tensor(array, near='c'):
    if isinstance(array, list):
        array = np.stack(array, axis=0)  # [t, h, w, c]
    if near == 'c':
        array = torch.from_numpy(np.transpose(array, (3, 0, 1, 2))).unsqueeze(0).float()  # [1, c, t, h, w]
    elif near == 't':
        array = torch.from_numpy(np.transpose(array, (0, 3, 1, 2))).unsqueeze(0).float() 
    else:
        raise ValueError(f'Unknown near type: {near}')
    return array


def tensor2np(array):
    array = torch.stack(array, dim=-1).squeeze(0).permute(1, 2, 0, 3).cpu().numpy()
    return array


def gradient_mask(mask):
    gradient_mask = np.logical_or.reduce((mask,
                                          np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)),
                                                         axis=0),
                                          np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)),
                                                         axis=1)))

    return gradient_mask


def indicesGen(pivot, interval, frames, t):
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


def get_ref_index(f, neighbor_ids, length, ref_length, num_ref):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


def save_flows(output, videoFlowF, videoFlowB):
    create_dir(os.path.join(output, 'completed_flow', 'forward_flo'))
    create_dir(os.path.join(output, 'completed_flow', 'backward_flo'))
    create_dir(os.path.join(output, 'completed_flow', 'forward_png'))
    create_dir(os.path.join(output, 'completed_flow', 'backward_png'))
    N = videoFlowF.shape[-1]
    for i in range(N):
        forward_flow = videoFlowF[..., i]
        backward_flow = videoFlowB[..., i]
        forward_flow_vis = cvbase.flow2rgb(forward_flow)
        backward_flow_vis = cvbase.flow2rgb(backward_flow)
        cvbase.write_flow(forward_flow, os.path.join(output, 'completed_flow', 'forward_flo', '{:05d}.flo'.format(i)))
        cvbase.write_flow(backward_flow, os.path.join(output, 'completed_flow', 'backward_flo', '{:05d}.flo'.format(i)))
        imageio.imwrite(os.path.join(output, 'completed_flow', 'forward_png', '{:05d}.png'.format(i)), forward_flow_vis)
        imageio.imwrite(os.path.join(output, 'completed_flow', 'backward_png', '{:05d}.png'.format(i)), backward_flow_vis)


def save_fgcp(output, frames, masks):
    create_dir(os.path.join(output, 'prop_frames'))
    create_dir(os.path.join(output, 'masks_left'))
    create_dir(os.path.join(output, 'prop_frames_npy'))
    create_dir(os.path.join(output, 'masks_left_npy'))

    assert len(frames) == masks.shape[2]
    for i in range(len(frames)):
        cv2.imwrite(os.path.join(output, 'prop_frames', '%05d.png' % i), frames[i] * 255.)
        cv2.imwrite(os.path.join(output, 'masks_left', '%05d.png' % i), masks[:, :, i] * 255.)
        np.save(os.path.join(output, 'prop_frames_npy', '%05d.npy' % i), frames[i] * 255.)
        np.save(os.path.join(output, 'masks_left_npy', '%05d.npy' % i), masks[:, :, i] * 255.)


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args, device):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model))

    model = model.module
    model.to(device)
    model.eval()

    return model


def initialize_LAFC(args, device):
    assert len(os.listdir(args.lafc_ckpts)) == 2
    checkpoint, config_file = glob.glob(os.path.join(args.lafc_ckpts, '*.tar'))[0], \
                              glob.glob(os.path.join(args.lafc_ckpts, '*.yaml'))[0]
    with open(config_file, 'r') as f:
        configs = yaml.full_load(f)
    model = configs['model']
    pkg = import_module('LAFC.models.{}'.format(model))
    model = pkg.Model(configs)
    state = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    return model, configs


def initialize_FGT(args, device, input_resolution=None):
    assert len(os.listdir(args.fgt_ckpts)) == 2
    checkpoint, config_file = glob.glob(os.path.join(args.fgt_ckpts, '*.tar'))[0], \
                              glob.glob(os.path.join(args.fgt_ckpts, '*.yaml'))[0]
    with open(config_file, 'r') as f:
        print(config_file)
        configs = yaml.full_load(f)

    if input_resolution is not None:
        configs['input_resolution'] = input_resolution

    model = configs['model']
    net = import_module('FGT.models.{}'.format(model))
    model = net.Model(configs).to(device)
    state = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(state['model_state_dict'])
    return model, configs


def calculate_flow(args, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW = args.imgH, args.imgW
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    if args.vis_flows:
        create_dir(os.path.join(args.outroot, 'flow', mode + '_flo'))
        create_dir(os.path.join(args.outroot, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            # resize optical flows
            h, w = flow.shape[:2]
            if h != imgH or w != imgW:
                flow = cv2.resize(flow, (imgW, imgH), cv2.INTER_LINEAR)
                flow[:, :, 0] *= (float(imgW) / float(w))
                flow[:, :, 1] *= (float(imgH) / float(h))
            
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            if args.vis_flows:
                # Flow visualization.
                flow_img = utils.flow_viz.flow_to_image(flow)
                flow_img = Image.fromarray(flow_img)

                # Saves the flow and flow_img.
                flow_img.save(os.path.join(args.outroot, 'flow', mode + '_png', '%05d.png' % i))
                utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow', mode + '_flo', '%05d.flo' % i), flow)

    return Flow


def extrapolation(args, video_ori, corrFlowF_ori, corrFlowB_ori):
    """Prepares the data for video extrapolation.
    """
    imgH, imgW, _, nFrame = video_ori.shape

    # Defines new FOV.
    imgH_extr = int(args.H_scale * imgH)
    imgW_extr = int(args.W_scale * imgW)
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Generates the mask for missing region.
    flow_mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.bool)
    flow_mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0

    mask_dilated = gradient_mask(flow_mask)

    # Extrapolates the FOV for video.
    video = np.zeros(((imgH_extr, imgW_extr, 3, nFrame)), dtype=np.float32)
    video[H_start: H_start + imgH, W_start: W_start + imgW, :, :] = video_ori

    for i in range(nFrame):
        print("Preparing frame {0}".format(i), '\r', end='')
        video[:, :, :, i] = cv2.inpaint((video[:, :, :, i] * 255).astype(np.uint8), flow_mask.astype(np.uint8), 3,
                                        cv2.INPAINT_TELEA).astype(np.float32) / 255.

    # Extrapolates the FOV for flow.
    corrFlowF = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowB = np.zeros(((imgH_extr, imgW_extr, 2, nFrame - 1)), dtype=np.float32)
    corrFlowF[H_start: H_start + imgH, W_start: W_start + imgW, :] = corrFlowF_ori
    corrFlowB[H_start: H_start + imgH, W_start: W_start + imgW, :] = corrFlowB_ori

    return video, corrFlowF, corrFlowB, flow_mask, mask_dilated, (W_start, H_start), (W_start + imgW, H_start + imgH)


def complete_flow(config, flow_model, flows, flow_masks, mode, device):
    if mode not in ['forward', 'backward']:
        raise NotImplementedError(f'Error flow mode {mode}')
    flow_masks = np.moveaxis(flow_masks, -1, 0)  # [N, H, W]
    flows = np.moveaxis(flows, -1, 0)  # [N, H, W, 2]
    if len(flow_masks.shape) == 3:
        flow_masks = flow_masks[:, :, :, np.newaxis]
    if mode == 'forward':
        flow_masks = flow_masks[0:-1]
    else:
        flow_masks = flow_masks[1:]

    num_flows, flow_interval = config['num_flows'], config['flow_interval']

    print('complete flow diffusion')
    diffused_flows = diffusion(flows, flow_masks)

    flows = np2tensor(flows)
    flow_masks = np2tensor(flow_masks)
    diffused_flows = np2tensor(diffused_flows)

    flows = flows.to(device)
    flow_masks = flow_masks.to(device)
    diffused_flows = diffused_flows.to(device)

    t = diffused_flows.shape[2]
    filled_flows = [None] * t
    pivot = num_flows // 2
    print('complete flow loop')
    for i in range(t):
        indices = indicesGen(i, flow_interval, num_flows, t)
        print('Indices: ', indices, '\r', end='')
        cand_flows = flows[:, :, indices]
        cand_masks = flow_masks[:, :, indices]
        inputs = diffused_flows[:, :, indices]
        pivot_mask = cand_masks[:, :, pivot]
        pivot_flow = cand_flows[:, :, pivot]
        with torch.no_grad():
            output_flow = flow_model(inputs, cand_masks)
        if isinstance(output_flow, tuple) or isinstance(output_flow, list):
            output_flow = output_flow[0]
        comp = output_flow * pivot_mask + pivot_flow * (1 - pivot_mask)
        if filled_flows[i] is None:
            filled_flows[i] = comp
    assert None not in filled_flows
    return filled_flows


def read_flow(flow_dir, video):
    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    flows = sorted(glob.glob(os.path.join(flow_dir, '*.flo')))
    for flow in flows:
        flow_data = cvbase.read_flow(flow)
        h, w = flow_data.shape[:2]
        flow_data = cv2.resize(flow_data, (imgW, imgH), cv2.INTER_LINEAR)
        flow_data[:, :, 0] *= (float(imgW) / float(w))
        flow_data[:, :, 1] *= (float(imgH) / float(h))
        Flow = np.concatenate((Flow, flow_data[..., None]), axis=-1)
    return Flow


def norm_flows(flows):
    assert len(flows.shape) == 5, 'FLow shape: {}'.format(flows.shape)
    flattened_flows = flows.flatten(3)
    flow_max = torch.max(flattened_flows, dim=-1, keepdim=True)[0]
    flows = flows / flow_max.unsqueeze(-1)
    return flows


def save_results(outdir, comp_frames):
    out_dir = os.path.join(outdir, 'frames')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(len(comp_frames)):
        if comp_frames[i] is not None:
            out_path = os.path.join(out_dir, '{:05d}.png'.format(i))
            print(comp_frames[i][:, :, ::-1].shape)
            print(comp_frames[i][:, :, ::-1].dtype)
            cv2.imwrite(out_path, comp_frames[i][:, :, ::-1].astype(np.uint8))


def video_inpainting(args):
#     device = torch.device('cuda:{}'.format(args.gpu))
    device = 'cuda:0'
    
    if args.opt is not None:
        with open(args.opt, 'r') as f:
            opts = yaml.full_load(f)

    for k in opts.keys():
        if k in args:
            setattr(args, k, opts[k])

    # Flow model.
    RAFT_model = initialize_RAFT(args, device)
    # LAFC (flow completion)
    LAFC_model, LAFC_config = initialize_LAFC(args, device)
    # FGT
#     FGT_model, FGT_config = initialize_FGT(args, device)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = args.imgH, args.imgW
    nFrame = len(filename_list)

    if imgH < 350:
        flowH, flowW = imgH * 2, imgW * 2
    else:
        flowH, flowW = imgH, imgW

    # Load video.
    video, video_flow = [], []
    for filename in sorted(filename_list):
        frame = torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float().unsqueeze(0)
        frame = F2.upsample(frame, size=(imgH, imgW), mode='bilinear', align_corners=False)
        frame_flow = F2.upsample(frame, size=(flowH, flowW), mode='bilinear', align_corners=False)
        video.append(frame)
        video_flow.append(frame_flow)

    video = torch.cat(video, dim=0)  # [n, c, h, w]
    video_flow = torch.cat(video_flow, dim=0)
    gts = video.clone()
    video = video.to('cuda')
    video_flow = video_flow.to(device)

    # Calcutes the corrupted flow.
    forward_flows = calculate_flow(args, RAFT_model, video_flow, 'forward')  # [B, C, 2, N]
    backward_flows = calculate_flow(args, RAFT_model, video_flow, 'backward')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.  # np array -> [h, w, c, N] (0~1)

    if args.mode == 'video_extrapolation':

        # Creates video and flow where the extrapolated region are missing.
        video, forward_flows, backward_flows, flow_mask, mask_dilated, start_point, end_point = extrapolation(args,
                                                                                                              video,
                                                                                                              forward_flows,
                                                                                                              backward_flows)
        imgH, imgW = video.shape[:2]

        # mask indicating the missing region in the video.
        mask = np.tile(flow_mask[..., None], (1, 1, nFrame))
        flow_mask = np.tile(flow_mask[..., None], (1, 1, nFrame))
        mask_dilated = np.tile(mask_dilated[..., None], (1, 1, nFrame))

    else:
        # Loads masks.
        filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                        glob.glob(os.path.join(args.path_mask, '*.jpg'))

        mask = []
        mask_dilated = []
        flow_mask = []
        for filename in sorted(filename_list):
            mask_img = np.array(Image.open(filename).convert('L'))
            mask_img = cv2.resize(mask_img, dsize=(imgW, imgH), interpolation=cv2.INTER_NEAREST)

            if args.flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.flow_mask_dilates)
            else:
                flow_mask_img = mask_img
            flow_mask.append(flow_mask_img)

            if args.frame_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.frame_dilates)
            mask.append(mask_img)
            mask_dilated.append(gradient_mask(mask_img))

        # mask indicating the missing region in the video.
        mask = np.stack(mask, -1).astype(np.bool)  # [H, W, C, N]
        mask_full = mask.copy()

        mask_dilated = np.stack(mask_dilated, -1).astype(np.bool)
        flow_mask = np.stack(flow_mask, -1).astype(np.bool)

    # Completes the flow.
    print('complete flow forward')
    videoFlowF = complete_flow(LAFC_config, LAFC_model, forward_flows, flow_mask, 'forward', device)
    print('complete flow backward')
    videoFlowB = complete_flow(LAFC_config, LAFC_model, backward_flows, flow_mask, 'backward', device)
    videoFlowF = tensor2np(videoFlowF)
    videoFlowB = tensor2np(videoFlowB)
    print('\nFinish flow completion.')

    if args.vis_completed_flows:
        save_flows(args.outroot, videoFlowF, videoFlowB)

    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)

    for indFrame in range(nFrame):
        img = video[:, :, :, indFrame]
        img[mask[:, :, indFrame], :] = 0
        img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3,
                          cv2.INPAINT_TELEA).astype(np.float32) / 255.

        gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)),
                                     axis=1)
        gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
        gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
        gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)

        gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0

    gradient_x_filled = gradient_x
    gradient_y_filled = gradient_y
    mask_gradient = mask_dilated
    video_comp = video

    # Gradient propagation.
    gradient_x_filled, gradient_y_filled, mask_gradient = \
        get_flowNN_gradient(args,
                            gradient_x_filled,
                            gradient_y_filled,
                            mask,
                            mask_gradient,
                            videoFlowF,
                            videoFlowB,
                            None,
                            None)

    # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
#     for indFrame in range(nFrame):
#         mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(
#             np.bool)

    # After one gradient propagation iteration
    # gradient --> RGB
    frameBlends = []
    
    for indFrame in range(nFrame):
        print("Poisson blending frame {0:3d}".format(indFrame))

        if mask[:, :, indFrame].sum() > 0:
#             try:
            start = time.time()
            frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame],
                                                         gradient_x_filled[:, 0: imgW - 1, :, indFrame],
                                                         gradient_y_filled[0: imgH - 1, :, :, indFrame],
                                                         mask[:, :, indFrame], mask_gradient[:, :, indFrame])

            print('poisson blend', time.time() - start)
            print(frameBlend.shape, UnfilledMask.shape)
#             except:
#                 frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]
#             print(frameBlend.shape, UnfilledMask.shape)

            frameBlend = np.clip(frameBlend, 0, 1.0)
            tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3,
                              cv2.INPAINT_TELEA).astype(np.float32) / 255.
            frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

            video_comp[:, :, :, indFrame] = frameBlend
            mask[:, :, indFrame] = UnfilledMask

            frameBlend_ = copy.deepcopy(frameBlend)
            # Green indicates the regions that are not filled yet.
            frameBlend_[mask[:, :, indFrame], :] = [0, 1., 0]
        else:
            frameBlend_ = video_comp[:, :, :, indFrame]
        frameBlends.append(frameBlend_)

    if args.vis_prop:
        save_fgcp(args.outroot, frameBlends, mask)
        
    videoFlowF = np.moveaxis(videoFlowF, -1, 0)
    videoFlowF = np.concatenate([videoFlowF, videoFlowF[-1:, ...]], axis=0)
    flows = np2tensor(videoFlowF, near='t')
    flows = norm_flows(flows).half()
    
    del videoFlowB
    
    torch.save(frameBlends, 'frameBlends_l.pt')
    torch.save(flows, 'flows_l.pt')
    torch.save(mask, 'mask_l.pt')
    torch.save(mask_full, 'mask_full.pt')

    
    frameBlends = torch.load('frameBlends.pt')
    flows = torch.load('flows.pt')
    mask = torch.load('mask.pt')
    mask_full = torch.load('mask_full.pt')

    frameBlends_l = torch.load('frameBlends_l.pt')
    flows_l = torch.load('flows_l.pt')
    mask_l = torch.load('mask_l.pt')

    del RAFT_model, LAFC_model
    torch.cuda.empty_cache()

    video_length = len(frameBlends)

    for i in range(len(frameBlends)):
        frameBlends[i] = frameBlends[i][:, :, ::-1]

    for i in range(len(frameBlends)):
        frameBlends_l[i] = frameBlends_l[i][:, :, ::-1]

    frames_first = np2tensor(frameBlends, near='t').half()
    frames_first_l = np2tensor(frameBlends_l, near='t').half()

    mask = np.moveaxis(mask, -1, 0)
    mask_l = np.moveaxis(mask_l, -1, 0)
    mask_full = np.moveaxis(mask_full, -1, 0)

    mask = mask[:, :, :, np.newaxis]
    mask_l = mask_l[:, :, :, np.newaxis]
    mask_full = mask_full[:, :, :, np.newaxis]

    masks = np2tensor(mask, near='t').half()
    masks_l = np2tensor(mask_l, near='t').half()
    masks_full = np2tensor(mask_full, near='t').half()

    normed_frames = frames_first * 2 - 1
    normed_frames_l = frames_first_l * 2 - 1

    comp_frames = [None] * video_length

    ref_length = args.step
    num_ref = args.num_ref
    neighbor_stride = args.neighbor_stride
    
    FGT_model, FGT_config = initialize_FGT(args, device)
    for p in FGT_model.parameters(): 
        p.requires_grad = False
    
    FGT_model = FGT_model.half()
    for f in range(0, video_length, neighbor_stride):
        
        f = 60 
        
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_length, num_ref)
        
        selected_frames = normed_frames[:, neighbor_ids + ref_ids]
        selected_frames_l = normed_frames_l[:, neighbor_ids + ref_ids]

        selected_masks = masks[:, neighbor_ids + ref_ids]
        selected_masks_l = masks_l[:, neighbor_ids + ref_ids]
        selected_masks_full = masks_full[:, neighbor_ids + ref_ids]

        masked_frames = selected_frames * (1 - selected_masks)
        masked_frames_l = selected_frames_l * (1 - selected_masks_l)
        masked_frames_full = selected_frames * (1 - selected_masks_full)

        selected_flows = flows[:, neighbor_ids + ref_ids]
        selected_flows_l = flows_l[:, neighbor_ids + ref_ids]
        # with torch.no_grad():
        
        filled_frames = hallucintaion1(
            masked_frames.float().to(device),
            masked_frames_l.float().to(device),
            selected_flows.float().to(device),
            selected_flows_l.float().to(device), 
            selected_masks.float().to(device),
            selected_masks_l.float().to(device),
            FGT_model.float(),
        ).cpu()
            
           
        del masked_frames, selected_flows, selected_masks
        torch.cuda.empty_cache()
        
        filled_frames = (filled_frames + 1) / 2
        filled_frames = filled_frames.permute(0, 2, 3, 1).numpy() * 255
        for i in range(len(neighbor_ids)):
            idx = neighbor_ids[i]
            valid_frame = frames_first[0, idx].cpu().permute(1, 2, 0).numpy() * 255.
            # valid_mask = masks[0, idx].cpu().permute(1, 2, 0).numpy()
            
            valid_mask = masks_full[0, idx].cpu().permute(1, 2, 0).numpy()
            comp = np.array(filled_frames[i]).astype(np.uint8) * valid_mask + \
                   np.array(valid_frame).astype(np.uint8) * (1 - valid_mask)
            if comp_frames[idx] is None:
                comp_frames[idx] = comp
            else:
                comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + comp.astype(np.float32) * 0.5
                
        break 
        
    if args.vis_frame:
        save_results(args.outroot, comp_frames)
    create_dir(args.outroot)
    
#     for i in range(len(comp_frames)):
#         comp_frames[i] = comp_frames[i].astype(np.uint8)
        
#     imageio.mimwrite(os.path.join(args.outroot, 'result_lama_refine.mp4'), comp_frames, fps=30, quality=8)
#     print(f'Done, please check your result in {args.outroot} ')

    
def downsample(frame, flow, mask, downscale_factor=0.5):
    shape = frame.shape
    h, w = shape[-2:]
    h, w = int(h * downscale_factor), int(w * downscale_factor)
    
    frame = F2.interpolate(frame, size=[frame.shape[-3], h, w])
    flow = F2.interpolate(flow, size=[flow.shape[-3], h, w])
    mask = F2.interpolate(mask, size=[mask.shape[-3], h, w])
    flow[:,:,0] /= 2
    flow[:,:,1] /= 2
    return frame, flow, mask


def downsample1(tensor, downscale_factor=0.5):
    shape = tensor.shape
    h, w = shape[-2:]
    h, w = int(h * downscale_factor), int(w * downscale_factor)
    
    tensor = F2.interpolate(tensor, size=[tensor.shape[-3], h, w])
    return tensor


def l1_masked(gt, pred, mask):
#     batch_size = gt.shape[0]
    
#     diff = ((gt - pred)*mask).abs()
#     diff = diff.view(batch_size, -1)

#     mask_size = mask.view(batch_size, -1)*3
#     loss = diff.sum(dim=1)/mask_size.sum(dim=1)
#     loss = diff.sum(dim=0)/mask_size.sum(dim=0)

    gt_masked = torch.masked_select(gt, mask)
    pred_masked = torch.masked_select(pred, mask)

    loss = F2.l1_loss(gt_masked, pred_masked)
    return loss
    

def refine(target, z, mask, model, n_iter=35, lr=1e-1):
    
    z = z.detach().cuda()

    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=lr)

    print(target.shape, mask.shape)
    mask = downsample1(mask)
    mask = mask >= 1
    for _ in range(n_iter):
        optimizer.zero_grad()

        pred = model.decode(z)
        pred = F2.interpolate(pred, size=list(target.shape[-2:]), mode='bilinear', align_corners=True)

        loss = l1_masked(target, pred, mask)
        print(_, loss.sum().item())
        
        if loss.sum().item() == 0:
            break
        
        loss.backward()
        
#         clip_value = 5
#         torch.nn.utils.clip_grad_norm_(z, clip_value)
       
        optimizer.step()
        
        del pred
        del loss
        
        torch.cuda.empty_cache()

    with torch.no_grad():
        result = model.decode(z).detach()
    return result


def hallucintaion(frame, flow, mask, model, n_iter=2):

    print('Hallucintaion stage')
    
    z_list, mask_list = [], []
    for i in range(n_iter):
        with torch.no_grad():
            z = model.cuda().half().encode(
                frame.cuda().half(),
                flow.cuda().half(),
                mask.cuda().half(),
                ).cpu()

            z_list.append(z)
            mask_list.append(mask)

            frame, flow, mask = downsample(frame, flow, mask, 0.5)
            
    z_list = z_list[::-1]
    mask_list = mask_list[::-1]

    model = model.double()

    for z in z_list:
        print('z list', torch.isnan(z).sum())
    
    with torch.no_grad():
        target = model.decode(z_list[0].double().cuda()).detach()

    lrs = [0.1, 0.1]
    for i in range(1, len(z_list)):
        print('level', i, tuple(mask_list[i-1].shape[-2:]), 'lr', lrs[i])
        z = z_list[i].double()
        mask = mask_list[i-1] >= 1
        target = refine(target.cuda(), z, mask.cuda(), model, lr=lrs[i])
    print(type(target))
    return target


def hallucintaion1(frame, frame_l, flow, flow_l, mask, mask_l, model):

    print('Hallucintaion stage')
    
    with torch.no_grad():
        z = model.cuda().encode(
            frame.cuda(),
            flow.cuda(),
            mask.cuda(),
            ).cpu()
    
    with torch.no_grad():
        pred  = model(frame_l, flow_l, mask_l)
        target = pred * mask_l + frame_l * (1 - mask_l)
                                                               
    target = refine(target, z, mask, model)
    
    return target


def main(args):
    assert args.mode in ('object_removal', 'video_extrapolation'), (
                                                                       "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
                                                                   ) % args.mode
    video_inpainting(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', default='configs/object_removal.yaml', help='Please select your config file for inference')
    # video completion
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='/myData/davis_resized/walking', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='/myData/dilateAnnotations_4/walking', help="mask for object removal")
    parser.add_argument('--outroot', default='quick_start/walking3', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=5, type=float,
                        help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # RAFT
    parser.add_argument('--raft_model', default='../LAFC/flowCheckPoint/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # LAFC
    parser.add_argument('--lafc_ckpts', type=str, default='../LAFC/checkpoint')

    # FGT
    parser.add_argument('--fgt_ckpts', type=str, default='../FGT/checkpoint')

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    # Image basic information
    parser.add_argument('--imgH', type=int, default=256)
    parser.add_argument('--imgW', type=int, default=432)
    parser.add_argument('--flow_mask_dilates', type=int, default=8)
    parser.add_argument('--frame_dilates', type=int, default=0)

    parser.add_argument('--gpu', type=int, default=0)

    # FGT inference parameters
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--num_ref', type=int, default=-1)
    parser.add_argument('--neighbor_stride', type=int, default=5)

    # visualization
    parser.add_argument('--vis_flows', action='store_true', help='Visualize the initialized flows')
    parser.add_argument('--vis_completed_flows', action='store_true', help='Visualize the completed flows')
    parser.add_argument('--vis_prop', action='store_true',
                        help='Visualize the frames after stage-I filling (flow guided content propagation)')
    parser.add_argument('--vis_frame', action='store_true', help='Visualize frames')

    args = parser.parse_args()

    main(args)

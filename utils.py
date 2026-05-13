import numpy as np
import cv2
import torch
from PIL import Image

def temporal_smooth(frames):
    """
    Applies optical flow-based temporal smoothing to a list of frames.
    Frames can be float32 [0,1], uint8 [0,255] numpy arrays, or PIL Images.
    Returns a list of uint8 [0,255] numpy arrays.
    """
    smoothed_frames_output = []

    # Handle input types and convert to float32 [0,1] for processing
    float_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            f_np = np.array(f).astype(np.float32) / 255.0
        elif f.dtype == np.uint8:
            f_np = f.astype(np.float32) / 255.0
        else:
            f_np = f
        float_frames.append(f_np)

    # Start with the first frame converted to uint8
    first_frame_uint8 = np.clip(float_frames[0] * 255, 0, 255).astype(np.uint8)
    smoothed_frames_output.append(first_frame_uint8)

    for i in range(1, len(float_frames)):
        prev_float = float_frames[i-1]
        curr_float = float_frames[i]

        prev_gray_uint8 = cv2.cvtColor(np.clip(prev_float * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        curr_gray_uint8 = cv2.cvtColor(np.clip(curr_float * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray_uint8, curr_gray_uint8, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        h, w = flow.shape[:2]
        # Generate coordinate maps for remap
        mapx, mapy = np.meshgrid(np.arange(w), np.arange(h))
        mapx = mapx.astype(np.float32) + flow[:,:,0]
        mapy = mapy.astype(np.float32) + flow[:,:,1]

        # Remap the current frame
        warped_float = cv2.remap(curr_float, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Blend the original current frame with the warped frame
        blended_float = (0.6 * curr_float + 0.4 * warped_float)

        # Convert the blended frame back to uint8
        blended_uint8 = np.clip(blended_float * 255, 0, 255).astype(np.uint8)
        smoothed_frames_output.append(blended_uint8)

    return smoothed_frames_output

def slerp(v0, v1, t, dot_threshold=0.9995):
    """
    Spherical Linear Interpolation (Slerp) smoothly interpolates between two high-dimensional noise vectors.
    """
    # Convert to float32 since numpy doesn't support bfloat16 natively
    v0_np = v0.detach().to(torch.float32).cpu().numpy()
    v1_np = v1.detach().to(torch.float32).cpu().numpy()
    dot = np.sum(v0_np * v1_np)

    if dot > dot_threshold:
        result = v0_np + t * (v1_np - v0_np)
    else:
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        result = s0 * v0_np + s1 * v1_np

    return torch.tensor(result, dtype=torch.float32, device=v0.device).to(v0.dtype)

import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def warp(img, flow):
    """
    Backward warp image using optical flow.
    img: [B, C, H, W]
    flow: [B, 2, H, W]
    """
    B, C, H, W = img.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(img.device)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)        
    output = F.grid_sample(img, vgrid, align_corners=True)
    return output

def ai_frame_interpolation(frames, device="cuda"):
    """
    Uses RAFT to synthesize high-quality intermediate frames.
    Input: list of PIL Images or numpy arrays.
    Returns: list of uint8 numpy arrays (doubled frame count).
    """
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    raft = raft_large(weights=weights, progress=False).to(device)
    raft.eval()
    
    # Standardize input to uint8 numpy
    uint8_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            uint8_frames.append(np.array(f))
        elif f.dtype == np.float32 or f.dtype == np.float64:
            uint8_frames.append((np.clip(f, 0, 1) * 255).astype(np.uint8))
        else:
            uint8_frames.append(f)
            
    output_frames = [uint8_frames[0]]
    
    with torch.no_grad():
        for i in range(len(uint8_frames) - 1):
            img1 = uint8_frames[i]
            img2 = uint8_frames[i+1]
            
            # RAFT expects tensors [1, 3, H, W] in float [-1, 1] via transforms
            img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
            img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
            
            img1_t, img2_t = transforms(img1_t, img2_t)
            
            # Compute flow
            flow_0_1 = raft(img1_t, img2_t)[-1]
            flow_1_0 = raft(img2_t, img1_t)[-1]
            
            # Warp for t=0.5
            # We scale the image to [0,1] for warping
            img1_w = img1_t / 255.0
            img2_w = img2_t / 255.0
            
            warp1 = warp(img1_w, flow_0_1 * 0.5)
            warp2 = warp(img2_w, flow_1_0 * 0.5)
            
            blended_t = 0.5 * warp1 + 0.5 * warp2
            blended_np = (blended_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            output_frames.append(blended_np)
            output_frames.append(img2)
            
    return output_frames

import os

def ai_upscale(frames, device="cuda"):
    """
    Uses Real-ESRGAN to upscale frames by 2x.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("RealESRGAN not installed. Skipping upscale.")
        return frames

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = '/home/novacare/selected-ramez/RealESRGAN_x4plus.pth'
    
    if not os.path.exists(model_path):
        print("RealESRGAN model weights not found! Skipping upscale.")
        return frames

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=torch.device(device)
    )
    
    upscaled_frames = []
    for f in frames:
        # Convert RGB to BGR for RealESRGAN
        img_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(img_bgr, outscale=2)
        # Convert back to RGB
        img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        upscaled_frames.append(img_rgb)
        
    return upscaled_frames

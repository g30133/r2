import io
import os
import sys
import time
import mmap
import threading

import torch as th
import numpy as np

import struct
import argparse
import re
import cv2
import json

from Xlib import X, display, error
from Xlib.ext import xfixes
from Xlib.ext.xtest import fake_input
from Xlib.XK import string_to_keysym
from Xlib.Xatom import WM_NAME

from steve1.config import PRIOR_INFO, DEVICE
from steve1.utils.mineclip_agent_env_utils import make_agent, load_mineclip_wconfig, load_model_parameters
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed

from detect_minecraft_menu import detect_letters

from steve1.VPT.agent import MineRLAgent, resize_image, AGENT_RESOLUTION, validate_env, \
     default_device_type, set_default_torch_device, CameraHierarchicalMapping, \
     ActionTransformer, ACTION_TRANSFORMER_KWARGS, POLICY_KWARGS, PI_HEAD_KWARGS
from steve1.embed_conditioned_policy import MinecraftAgentPolicy

from gym3.types import DictType

POLL_INTERVAL_PROMPT = 0.1
POLL_INTERVAL_FRAME = 0.001
WIDTH = 640
HEIGHT = 360
FRAMETYPE_GAMEPLAY = 0
FRAMETYPE_GAMEMENU = 1
HEADER_SIZE = 4 + 4 + 4
FRAME_SIZE = 4 + 4 + WIDTH * HEIGHT * 4
CONTROL_SIZE = 4 + 4 + 4
PROMPT_SIZE = 160
SHMBUF_FRAME_SIZE = HEADER_SIZE + FRAME_SIZE
SHMBUF_FRAME_PATH = "/dev/shm/minecraft_frame"
SHMBUF_CONTROL_SIZE = HEADER_SIZE + CONTROL_SIZE
SHMBUF_CONTROL_PATH = "/dev/shm/minecraft_control"
SHMBUF_PROMPT_SIZE = HEADER_SIZE + PROMPT_SIZE
SHMBUF_PROMPT_PATH = "/dev/shm/minecraft_prompt"

DEFAULT_PROMPT = 'gather wood'

# scaler to convert from pixel delta to angle delta
# taken from Video Pre-Training's run_inverse_dynamics_model.py
CAMERA_SCALER = 360.0 / 2400.0
EXTRA_FACTOR = 1.0

def draw_cursor(frame, ctx, x, y):
    
    cursor_bgr = ctx['cursor_bgr']
    cursor_alpha = ctx['cursor_alpha']

    ch = ctx['ch']
    cw = ctx['cw']
    h, w = frame.shape[:2]

    if x < 0 or y < 0 or x + cw > w or y + ch > h:
        return  # Skip drawing if out of bounds

    roi = frame[y:y+ch, x:x+cw]

    for c in range(3):
        roi[:, :, c] = (cursor_alpha * cursor_bgr[:, :, c] +
                        (1 - cursor_alpha) * roi[:, :, c]).astype(np.uint8)

    frame[y:y+ch, x:x+cw] = roi


def get_action(frame, ctx):
    with th.cuda.amp.autocast():
        device = DEVICE

        agent_input = resize_image(frame, AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(device)}
        agent_input['mineclip_embed'] = th.from_numpy(ctx['prompt_embed']).to(device)
        #print("get_action agent_input['img'].shape:", agent_input['img'].shape)
        agent_action, hidden_state, result = ctx['policy'].act(
            agent_input, ctx['_dummy_first'], ctx['hidden_state'],
            stochastic=True, cond_scale=ctx['cond_scale'])
        ctx['hidden_state'] = hidden_state

        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        #print('agent_action:', agent_action)
        #print('action:', action)

        action0 = ctx['action_mapper'].to_factored(action)
        return action0
  
def init_policy(in_model, in_weights, prior_info, cond_scale):
    print('====init_policy====')

    device = DEVICE

    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    #action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    policy_kwargs, pi_head_kwargs = load_model_parameters(in_model)
    if policy_kwargs is None: policy_kwargs = POLICY_KWARGS
    if pi_head_kwargs is None: pi_head_kwargs = PI_HEAD_KWARGS

    action_space = action_mapper.get_action_space_update()
    print('action_space:', action_space)
    action_space = DictType(**action_space)
    print('action_space:', action_space)

    agent_kwargs = dict(
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
        action_space=action_space)
    policy = MinecraftAgentPolicy(**agent_kwargs).to(device)

    _dummy_first = th.from_numpy(np.array((False,))).to(device)
    hidden_state = policy.initial_state(1 if cond_scale == None else 2)

    """Load model weights from a path, and reset hidden state"""
    state_dict = th.load(in_weights, map_location=device)
    state_dict = {k.replace("module.policy.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=False)

    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(prior_info)

    print('====init_policy DONE====')
    return policy, _dummy_first, hidden_state, action_mapper, mineclip, prior


def extract_command_json(text):
    # Case A: full JSON object
    if prompt_str.startswith('{'):
        try:
            return json.loads(prompt_str)["command"].strip()
        except (json.JSONDecodeError, KeyError):
            return None

    # Case B: already a plain command
    return prompt_str            # treat whole string as the command

def process_prompt(ctx):
    shmbuf_prompt = ctx['shmbuf_prompt']

    promptcounter = 0

    while True:
        time.sleep(POLL_INTERVAL_PROMPT)

        shmbuf_prompt.seek(0)
        header0 = struct.unpack("<I", shmbuf_prompt.read(4))[0]

        if header0 == promptcounter:
            continue
        elif header0 > promptcounter + 1:
            print('header0:', header0, 'promptcounter:', promptcounter)

        promptcounter = header0
        
        shmbuf_prompt.seek(4)
        header4 = struct.unpack("<I", shmbuf_prompt.read(4))[0]
        promptlen = header4
    
        shmbuf_prompt.seek(HEADER_SIZE)
        raw_bytes = shmbuf_prompt.read(promptlen)
        prompt_str = raw_bytes.rstrip(b'\x00').decode('utf-8')

        if prompt_str[0] == '/':
            cheat = prompt_str[1:]
            # camera:[factor]
            # ex. camera:1.5, camera:0.7
            parts = cheat.split(':')
            if parts[0] == 'camera':
                ctx["camera_factor"] = float(parts[1])
                print(f"ctx['camera_factor']:{ctx['camera_factor']}")
        else:
            '''
            ctx["prompt_embed"] = get_prior_embed(prompt_str, ctx["mineclip"], ctx["prior"], DEVICE)
            ctx["prompt"] = prompt_str
            print(f"prompt:|{ctx['prompt']}|")
            '''
            cmd = prompt_str.strip()
            ctx["prompt"] = cmd
            ctx["prompt_embed"] = get_prior_embed(cmd, ctx["mineclip"], ctx["prior"], DEVICE)
            print("prompt:|", cmd, "|")

def init_prompt(ctx):
    print('====init_prompt====')
    
    with open(SHMBUF_PROMPT_PATH, "w+b") as f:
        f.truncate(SHMBUF_PROMPT_SIZE)
        shmbuf_prompt = mmap.mmap(f.fileno(), SHMBUF_PROMPT_SIZE, access=mmap.ACCESS_READ)

    ctx['shmbuf_prompt'] = shmbuf_prompt

    prompt_thread = threading.Thread(target=lambda : process_prompt(ctx))
    prompt_thread.daemon = True

    ctx['prompt'] = DEFAULT_PROMPT
    ctx["prompt_embed"] = get_prior_embed(DEFAULT_PROMPT, ctx["mineclip"], ctx["prior"], DEVICE)
    ctx['camera_factor'] = 1.0
    
    prompt_thread.start()
    
    print('====init_prompt DONE====')
    return prompt_thread


def process_frame_control(ctx):
    shmbuf_frame = ctx['shmbuf_frame']
    shmbuf_control = ctx['shmbuf_control']
    
    framecounter = 0

    while True:
        time.sleep(POLL_INTERVAL_FRAME)
        t1 = time.time()

            
        shmbuf_frame.seek(0)
        header0 = struct.unpack(">I", shmbuf_frame.read(4))[0]
        if header0 == framecounter:
            continue
        elif header0 > framecounter + 1:
            print('header0:', header0, 'framecounter:', framecounter)

        framecounter = header0
        shmbuf_frame.seek(4)
        header4 = struct.unpack(">I", shmbuf_frame.read(4))[0]
        frametype = header4
        
        ctx['framecounter'] = framecounter
        ctx['frametype'] = frametype

        shmbuf_frame.seek(HEADER_SIZE)
        mouseX = struct.unpack(">I", shmbuf_frame.read(4))[0]

        shmbuf_frame.seek(4 + HEADER_SIZE)
        mouseY = struct.unpack(">I", shmbuf_frame.read(4))[0]

        
        shmbuf_frame.seek(4 + 4 + HEADER_SIZE)
        frame_rgba = np.frombuffer(shmbuf_frame.read(FRAME_SIZE - 8), dtype=np.uint8)

        frame_rgb = frame_rgba.reshape((HEIGHT, WIDTH, 4))[:,:,:3]
        frame_rgb_flip = frame_rgb[::-1,:,:].copy()
        frame_bgr_flip = frame_rgb_flip[:,:,::-1]

        if frametype != FRAMETYPE_GAMEPLAY:
            draw_cursor(frame_rgb_flip, ctx, mouseX-6, mouseY-4) # offset since head of pointer not exactly top left 
        
        action0 = get_action(frame_rgb_flip, ctx)
        
        butval = action0['buttons'][0]
        camval0 = action0['camera'][0][0]
        camval1 = action0['camera'][0][1]
        
        button_control = 0
        for bit in butval:
            button_control = (button_control << 1) | bit
        camera_control = camval0 * 11 + camval1
        camera_factor = ctx['camera_factor']
        
        shmbuf_control.seek(HEADER_SIZE)
        shmbuf_control.write(struct.pack(">IIf", button_control, camera_control, camera_factor))
        shmbuf_control.seek(4)
        shmbuf_control.write(struct.pack(">I", frametype))
        shmbuf_control.flush()

        shmbuf_control.seek(0)
        shmbuf_control.write(struct.pack(">I", framecounter))
        shmbuf_control.flush()
        
        t2 = time.time()
        filename = f"{framecounter:06d} {frametype} [{butval[0]}{butval[1]}{butval[2]}{butval[3]}{butval[4]}|{butval[5]}{butval[6]}{butval[7]}{butval[8]}{butval[9]}|{butval[10]}{butval[11]}{butval[12]}{butval[13]}{butval[14]}|{butval[15]}{butval[16]}{butval[17]}{butval[18]}{butval[19]}][{camval0:02d}|{camval1:02d}] {ctx['camera_factor']:01.1f} {ctx['prompt']}"
        print(filename)
        #cv2.imwrite(f"/tmp/{filename}.png", frame_bgr_flip)

def init_frame_control(ctx):
    print('====init_frame_control====')

    with open(SHMBUF_FRAME_PATH, 'w+b') as f:
        f.truncate(SHMBUF_FRAME_SIZE)
        shmbuf_frame = mmap.mmap(f.fileno(), SHMBUF_FRAME_SIZE, access=mmap.ACCESS_READ)

    with open(SHMBUF_CONTROL_PATH, 'w+b') as f:
        f.truncate(SHMBUF_CONTROL_SIZE)
        shmbuf_control = mmap.mmap(f.fileno(), SHMBUF_CONTROL_SIZE, access=mmap.ACCESS_WRITE)
    
    ctx['shmbuf_frame'] = shmbuf_frame
    ctx['shmbuf_control'] = shmbuf_control

    frame_control_thread = threading.Thread(target=lambda : process_frame_control(ctx))
    frame_control_thread.daemon = True

    frame_control_thread.start()

    print('====init_frame_control DONE====')
    return frame_control_thread

def dumpBinIx2Pixels(agent):
    print('-----------------------------')
    for ix in range(11):
        pixel = binIx2pixel(ix, agent)
        print(f"{ix} {pixel}")
    print('-----------------------------')

    
def main(in_model, in_weights, prior_info, cond_scale):
    policy, _dummy_first, hidden_state, action_mapper, mineclip, prior = init_policy(in_model, in_weights, prior_info, cond_scale)

    #dumpBinIx2Pixels(agent);

    cursor = cv2.imread("/home/ubuntu/steve-1/steve1/run_agent/arrow_000.png", cv2.IMREAD_UNCHANGED)
    print("cursor:", cursor.shape)
    # Split into BGR
    cursor_bgr = cursor[:, :, :3]
    cursor_alpha = cursor[:, :, 3] / 255.0
    # Get height and width of the cursor
    ch, cw = cursor_bgr.shape[:2]

    
    ctx = {
        'policy': policy, '_dummy_first': _dummy_first, 'hidden_state': hidden_state,
        'action_mapper': action_mapper,
        'mineclip':mineclip, 'prior':prior, 'cond_scale': cond_scale,
        'cursor_bgr': cursor_bgr, 'cursor_alpha': cursor_alpha, 'ch': ch, 'cw': cw,
    }

    prompt_thread = init_prompt(ctx)
    frame_control_thread = init_frame_control(ctx)

    try:
        prompt_thread.join()
        frame_control_thread.join()
    except KeyboardInterrupt:
        pass

    # Clean up

def test_draw_cursor():
    frame = cv2.imread('./mc_frame.png')

    cursor = cv2.imread("./arrow_000.png", cv2.IMREAD_UNCHANGED)
    print(cursor.shape)
    cursor_bgr = cursor[:, :, :3]
    cursor_alpha = cursor[:, :, 3] / 255.0  # normalize alpha
    # Get height and width of the cursor
    ch, cw = cursor_bgr.shape[:2]
    
    ctx = {
        'cursor_bgr': cursor_bgr, 'cursor_alpha': cursor_alpha, 'ch': ch, 'cw': cw,
    }
    
    draw_cursor(frame, ctx, 100, 100)
    cv2.imwrite('./mc_frame_with_cursor.png', frame)



if __name__ == '__main__':
#    test_draw_cursor()

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--cond_scale', type=float, default=6.0)
    args = parser.parse_args()

    main(
        in_model=args.in_model,
        in_weights=args.in_weights,
        prior_info=PRIOR_INFO,
        cond_scale=args.cond_scale,
    )













    

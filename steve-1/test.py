import mmap
import numpy as np
import torch
import cv2
import time
import struct

POLL_INTERVAL = 0.01
WIDTH = 640
HEIGHT = 360
FRAME_SIZE = WIDTH * HEIGHT * 4
HEADER_SIZE = 4
SHMBUF_SIZE = HEADER_SIZE + FRAME_SIZE
SHMBUF_PATH = "/dev/shm/minecraft_frame"

# Load model
#model = torch.jit.load("minecraft_ai_model.pt").cuda()
#model.eval()

# Open shared memory
with open(SHMBUF_PATH, "r+b") as f:
    shmbuf = mmap.mmap(f.fileno(), SHMBUF_SIZE, access=mmap.ACCESS_READ)

framecounter = 0
while True:
    time.sleep(POLL_INTERVAL)

    shmbuf.seek(0)
    header = struct.unpack(">I", shmbuf.read(4))[0]
    if header == framecounter:
        print('.', end='')
        continue

    framecounter = header

    shmbuf.seek(HEADER_SIZE)
    frame_rgba = np.frombuffer(shmbuf.read(FRAME_SIZE), dtype=np.uint8)

    frame_rgb = frame_rgba.reshape((HEIGHT, WIDTH, 4))[:,:,:3]
    frame_bgr = frame_rgb[:,:,::-1] # rgb bgr switch
    frame_bgr_flip = frame_bgr[::-1,:,:] # up down switch

    cv2.imwrite(f'{framecounter:06d}frame.png', frame_bgr_flip)
    print(f"framecounter:{framecounter:06d}")


    # Preprocess for model
    #input_tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0

    # Run inference (async for max speed)
    #with torch.no_grad():
    #    controls = model(input_tensor)

    # Convert to key/mouse events
    #keypress, mouse_move = process_output(controls)

    # Send back to Forge mod
    #send_controls_to_minecraft(keypress, mouse_move)


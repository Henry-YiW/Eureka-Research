#!/bin/bash
export LD_LIBRARY_PATH=/home/jensenyuan/miniconda3/envs/eureka/lib:$LD_LIBRARY_PATH
python train.py \
    task=ShadowHandRope \
    #checkpoint='../../eureka/checkpoints/last_ShadowHandRope_ep_2000.pth' \
    checkpoint='/home/jensenyuan/Projects/Eureka-Research/isaacgymenvs/isaacgymenvs/outputs/train/2025-06-07_20-27-26/runs/ShadowHandRope-2025-06-07_20-27-26/nn/ShadowHandRope.pth' \
    test=True \
    headless=False \
    # force_render is not used by the rl_games player loop.
    # Rendering is enabled by headless=False, which sets player.render=True in train.py.
    +weights_only=True
#!/bin/bash
export LD_LIBRARY_PATH=/home/jensenyuan/miniconda3/envs/eureka/lib:$LD_LIBRARY_PATH
python train.py \
    task=ShadowHandSpin \
    #checkpoint='../../eureka/checkpoints/last_ShadowHandRope_ep_2000.pth' \
    checkpoint='/home/jensenyuan/Projects/Eureka-Research/eureka/checkpoints/ShadowHandSpinGPT_final_06-35-32.pth' \
    test=True \
    headless=False \
    force_render=True \
    # force_render is not used by the rl_games player loop.
    # Rendering is enabled by headless=False, which sets player.render=True in train.py.
    +weights_only=True
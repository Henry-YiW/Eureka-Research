python train.py \
    task=ShadowHandRope \
    checkpoint='../../eureka/checkpoints/last_ShadowHandRope_ep_2000.pth' \
    test=True \
    headless=False \
    force_render=True \
    +weights_only=True
import torch

checkpoint_path = '/scratch/bdes/haorany7/Eureka-Research/isaacgymenvs/isaacgymenvs/outputs/train/2025-05-13_03-49-14/runs/ShadowHandRope-2025-05-13_03-49-57/nn/last_ShadowHandRope_ep_2000.pth'

print(f"Attempting to load: {checkpoint_path}")
try:
    # Attempt to load with weights_only=True, which is safer
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    print("Successfully loaded checkpoint with weights_only=True.")
    print("Keys in checkpoint:", checkpoint_data.keys() if isinstance(checkpoint_data, dict) else "Checkpoint is not a dict")

    # If the above fails or if rl_games absolutely needs weights_only=False (unlikely for just agent weights)
    # you can try with weights_only=False, but be aware of the warning.
    # print("\nAttempting to load with weights_only=False (as a fallback test):")
    # checkpoint_data_unsafe = torch.load(checkpoint_path, map_location='cpu', weights_only=False) # original pickle behavior
    # print("Successfully loaded checkpoint with weights_only=False.")
    # print("Keys in checkpoint (unsafe load):", checkpoint_data_unsafe.keys() if isinstance(checkpoint_data_unsafe, dict) else "Checkpoint is not a dict")

except Exception as e:
    print(f"Error loading checkpoint directly: {e}")
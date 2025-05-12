import torch
import os # Import os to check if file exists

# --- UPDATE THIS LINE WITH THE CORRECT RELATIVE PATH ---
checkpoint_path = 'outputs/eureka/2025-05-12_04-07-53/policy-2025-05-12_05-05-34/runs/AntGPT-2025-05-12_05-05-35/nn/last_AntGPT_ep_3000.pth'
# -----------------------------------------------------

print(f"Attempting to load: {checkpoint_path}")

# Check if file exists before trying to load
if not os.path.exists(checkpoint_path):
    print(f"-----> ERROR: File does not exist at path: {checkpoint_path} <-----")
    print("Please double-check the path relative to your current directory.")
else:
    print("File found. Attempting torch.load...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("-----> Checkpoint loaded successfully on CPU! <-----")
        print("This means the file itself is readable by torch.load.")
        if isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary. Keys:", checkpoint.keys())
        else:
            print("Checkpoint is not a dictionary. Type:", type(checkpoint))

    except Exception as e:
        print(f"-----> Error loading checkpoint: {e} <-----")
        print("This suggests the checkpoint file might be corrupted, incomplete, or in an unexpected format.")

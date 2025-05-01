import isaacgym
from isaacgym import gymapi
import isaacgymenvs
import torch
import numpy as np
import os

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# Use PhysX
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.friction_offset_threshold = 0.01
sim_params.physx.friction_correlation_distance = 0.005
sim_params.physx.use_gpu = True # Make sure GPU is available

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params) # compute_device_id=0, graphics_device_id=0

if sim is None:
    print("*** Failed to create sim")
    quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up
plane_params.distance = 0
plane_params.static_friction = 0.5
plane_params.dynamic_friction = 0.5
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

# --- Asset Loading ---
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaacgymenvs/assets/mjcf/solid_rope")
asset_file = "rope.urdf" # Assuming your URDF file is named rope.urdf

if not os.path.exists(os.path.join(asset_root, asset_file)):
    print(f"*** Error: Asset file not found at {os.path.join(asset_root, asset_file)}")
    quit()

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False # Keep it false to see if gravity affects it, set true to fix it
asset_options.disable_gravity = False
# asset_options.use_mesh_materials = True # Uncomment if you want to use materials defined in the mesh files (e.g., MTL)

print(f"Loading asset '{asset_file}' from '{asset_root}'")
rope_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

if rope_asset is None:
    print("*** Failed to load asset")
    quit()

# --- Environment Setup ---
num_envs = 1
envs_per_row = 1
env_spacing = 1.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# Actor initial pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 1.0) # Initial position (1m above ground)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # Initial orientation

envs = []
actor_handles = []

print(f"Creating {num_envs} environments")
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # add actor
    actor_handle = gym.create_actor(env, rope_asset, pose, "rope", i, 0, 0) # collision group 0, segmentation id 0
    actor_handles.append(actor_handle)

# --- Viewer Setup ---
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Point camera at the environment
cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
cam_pos = gymapi.Vec3(2.0, 2.0, 2.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# --- Simulation Loop ---
print("Running simulation... Press Esc to exit.")
while not gym.query_viewer_has_closed(viewer):

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt
    gym.sync_frame_time(sim)

print("Simulation finished.")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim) 
# Adapted from shadow_hand_spin.py for ShadowHandRope task

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

# Note: compute_reward function is intentionally left minimal
# Eureka will generate the specific reward function logic

class ShadowHandRope(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.randomize = self.cfg["task"]["randomize"]
        if self.randomize:
            self.randomization_params = self.cfg["task"]["randomization_params"]
        else:
            self.randomization_params = None
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Reward scales - These might be placeholders if Eureka defines its own scales
        self.dist_reward_scale = self.cfg["env"].get("distRewardScale", 1.0)
        self.rot_reward_scale = self.cfg["env"].get("rotRewardScale", 1.0)
        self.action_penalty_scale = self.cfg["env"].get("actionPenaltyScale", 1.0)
        self.success_tolerance = self.cfg["env"].get("successTolerance", 0.02)
        self.reach_goal_bonus = self.cfg["env"].get("reachGoalBonus", 10.0)
        self.fall_dist = self.cfg["env"].get("fallDistance", 0.5)
        self.fall_penalty = self.cfg["env"].get("fallPenalty", -20.0)
        self.rot_eps = self.cfg["env"].get("rotEps", 0.1)

        # *** NEW for Rope Task ***
        self.target_displacement_positive = self.cfg["env"].get("targetDisplacementPositive", 0.1) # Example default value
        # *** End NEW ***

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        # *** Changed to handle 'rope' object type ***
        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type == "rope"

        # Paths for shadow hand and the rope
        self.asset_files_dict = {
            "hand": "mjcf/open_ai_assets/hand/shadow_hand.xml",
            "rope": "mjcf/open_ai_assets/hand/rope.xml" # Default path, can be overridden by config
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["hand"] = self.cfg["env"]["asset"].get("assetFileName", self.asset_files_dict["hand"])
            self.asset_files_dict["rope"] = self.cfg["env"]["asset"].get("assetFileNameRope", self.asset_files_dict["rope"])

        # Observation type
        self.obs_type = self.cfg["env"]["observationType"]
        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception("Unknown type of observations!")
        print("Obs type:", self.obs_type)

        # Observation dimension based on type
        self.num_obs_dict = {
            "openai": 44,
            "full_no_vel": 68,
            "full": 148,
            "full_state": 202
        }

        self.up_axis = 'z'
        self.up_axis_idx = 2

        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False # Based on ShadowHandSpin
        self.fingertip_obs = True # Based on ShadowHandSpin
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = self.num_obs_dict["full_state"] # Assuming full state used for asymmetry

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 20 # Shadow Hand standard actions

        # Call VecTask __init__
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # Setup camera
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.25, -0.8, 1.05)
            cam_target = gymapi.Vec3(-0.05, -0.2, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Acquire tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Sensor setup (if needed for observation type)
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Create wrapper tensors
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1) # Adjusted for hand + rope
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.local_y_axis = self.y_unit_tensor.clone() # Assuming rope primary axis is Y

        # Buffers - Note: Rope task might not need goal resetting logic like Spin
        # self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        # <<< Add reset_goal_buf initialization >>>
        self.reset_goal_buf = torch.zeros_like(self.reset_buf)
        # <<< End add reset_goal_buf initialization >>>

        # Random force parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # Target displacement state
        self.target_displacement_positive_val = self.target_displacement_positive # Store the initial positive value
        # Remove self.local_rope_end_offset initialization
        # self.local_rope_end_offset = torch.full((self.num_envs, 1), self.target_displacement_positive_val, dtype=torch.float, device=self.device)

        # Explicitly declare current_palm_displacement here for clarity
        self.current_palm_displacement = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Explicitly declare target_displacement for clarity and LLM interpretation
        # Initialize as a 1D tensor directly with the positive value
        self.target_displacement = torch.full((self.num_envs,), self.target_displacement_positive_val, dtype=torch.float, device=self.device)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Use asset paths from self.asset_files_dict
        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        shadow_hand_asset_file = self.asset_files_dict["hand"]
        rope_asset_file = self.asset_files_dict["rope"]

        # Load shadow hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE # Driven by position targets
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        # Get hand properties
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        # Set up tendon properties (copied from Spin)
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        for i in range(num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = to_torch([self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names], dtype=torch.long, device=self.device)

        # Store DOF limits and defaults
        self.shadow_hand_dof_lower_limits = to_torch([prop for prop in shadow_hand_dof_props['lower']], device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch([prop for prop in shadow_hand_dof_props['upper']], device=self.device)
        self.shadow_hand_dof_default_pos = torch.zeros(self.num_shadow_hand_dofs, device=self.device)
        self.shadow_hand_dof_default_vel = torch.zeros(self.num_shadow_hand_dofs, device=self.device)

        # Fingertip handles and sensors
        self.fingertip_handles = to_torch([self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips], dtype=torch.long, device=self.device)
        self.palm_handle = self.gym.find_asset_rigid_body_index(shadow_hand_asset, "robot0:palm")
        if self.palm_handle == -1:
            print("*** ERROR: Failed to find palm rigid body handle!")
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        # *** Load rope asset ***
        rope_asset_options = gymapi.AssetOptions()
        rope_asset_options.fix_base_link = False # Rope should be movable
        rope_asset_options.disable_gravity = False # Rope should fall
        rope_asset_options.use_mesh_materials = True # Use material from MTL file
        rope_asset_options.vhacd_enabled = True # Enable convex decomposition for better collision
        rope_asset_options.vhacd_params = gymapi.VhacdParams()
        rope_asset_options.vhacd_params.resolution = 100000
        rope_asset = self.gym.load_asset(self.sim, asset_root, rope_asset_file, rope_asset_options)
        rope_rb_count = self.gym.get_asset_rigid_body_count(rope_asset)

        # Define start poses
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)

        rope_start_pose = gymapi.Transform()
        rope_start_pose.p = shadow_hand_start_pose.p + gymapi.Vec3(0.0, -0.41, 0.15)  # Same as hand
        rope_start_pose.r = gymapi.Quat(0, 0, 0, 0.5)
        # Compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + rope_rb_count
        max_agg_shapes = self.num_shadow_hand_shapes + self.gym.get_asset_rigid_shape_count(rope_asset)

        # Initialize lists for envs and actors
        self.shadow_hands = []
        self.envs = []
        self.rope_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.rope_indices = []
        self.object_rb_handles = [] # Store rope rigid body handles
        shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        rope_rb_count = self.gym.get_asset_rigid_body_count(rope_asset)
        self.rope_rb_handles = list(range(shadow_hand_rb_count, shadow_hand_rb_count + rope_rb_count))

        # Create environments
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Add hand actor
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0]) # Zero initial velocity
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # Add rope actor
            rope_handle = self.gym.create_actor(env_ptr, rope_asset, rope_start_pose, "rope", i, 0, 0)
            rope_idx = self.gym.get_actor_index(env_ptr, rope_handle, gymapi.DOMAIN_SIM)
            
            # self._create_fixed_joint_between(env_ptr, shadow_hand_actor, rope_handle)
            
            self.rope_indices.append(rope_idx)
            self.rope_init_state.append([rope_start_pose.p.x, rope_start_pose.p.y, rope_start_pose.p.z,
                                          rope_start_pose.r.x, rope_start_pose.r.y, rope_start_pose.r.z, rope_start_pose.r.w,
                                          0, 0, 0, 0, 0, 0]) # Zero initial velocity

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # Store initial states and handles as tensors
        self.rope_init_state = to_torch(self.rope_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.rope_indices = to_torch(self.rope_indices, dtype=torch.long, device=self.device)
        self.rope_rb_handles = to_torch(self.rope_rb_handles, dtype=torch.long, device=self.device)

        # Get rope mass for random force calculation
        rope_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, rope_handle)
        self.rope_rb_masses = to_torch([prop.mass for prop in rope_rb_props], dtype=torch.float, device=self.device)
    
    def _create_fixed_joint_between(self, env_ptr, parent_actor, child_actor):
        # Get transforms of parent and child roots
        parent_pose = gymapi.Transform()
        child_pose = gymapi.Transform()

        self.gym.create_joint(
            env_ptr,
            parent_actor,
            child_actor,
            gymapi.JointType.JOINT_FIXED,
            parent_pose,
            child_pose
        )

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.current_palm_displacement, self.target_displacement,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist,
            self.fall_penalty, self.max_consecutive_successes, self.av_factor
        )
        # --- Removed intermediate variables and separate assignments --- 

        # --- Update extras (optional) ---
        self.extras["current_displacement"] = self.current_palm_displacement.mean()
        self.extras["target_displacement"] = self.target_displacement.mean()
        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        # Ensure gpt_reward exists if other parts of the system expect it (can be zero)
        if "gpt_reward" not in self.extras:
             self.extras["gpt_reward"] = torch.zeros_like(self.rew_buf).mean()

        # *** Print displacement error (loss) - Mean over envs ***
        displacement_error = torch.abs(self.current_palm_displacement - self.target_displacement)
        print(f"Mean Displacement Error: {displacement_error.mean().item():.4f}")
        # *** Print total reward (mean over envs) ***
        print(f"Mean Total Reward: {self.rew_buf.mean().item():.4f}")
        # <<< End: Modify compute_reward to call JIT function >>>

    def compute_observations(self):
        # Refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # Extract rope states
        # *** Renamed from object_... to rope_... ***
        self.rope_pose = self.root_state_tensor[self.rope_indices, 0:7]
        self.rope_pos = self.root_state_tensor[self.rope_indices, 0:3]
        self.rope_rot = self.root_state_tensor[self.rope_indices, 3:7]
        self.rope_linvel = self.root_state_tensor[self.rope_indices, 7:10]
        self.rope_angvel = self.root_state_tensor[self.rope_indices, 10:13]

        # Extract fingertip states
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # *** Calculate palm displacement along rope axis ***
        palm_pos = self.rigid_body_states[:, self.palm_handle, 0:3]
        rope_y_axis_world = quat_apply(self.rope_rot, self.local_y_axis)
        vec_rope_to_palm = palm_pos - self.rope_pos
        self.current_palm_displacement = torch.sum(vec_rope_to_palm * rope_y_axis_world, dim=1)
        # *** End Calculate palm displacement ***

        # Select observation computation based on type
        # Note: Observation buffers might need size adjustments based on rope state inclusion
        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        else:
            print("Unknown observations type!")

        # Compute asymmetric observations if needed
        if self.asymmetric_obs:
            self.compute_full_state(True)

    # Observation helper functions (Adapted from ShadowHandSpin)
    # These might need modification to include relevant rope information
    def compute_fingertip_observations(self, no_vel=False):
        # Placeholder - Needs modification for rope task
        if no_vel:
            # Use pre-calculated palm displacement
            current_displacement = self.current_palm_displacement

            # Revised structure:
            # Fingertip position (15)
            # Rope position (3)
            # Rope rotation (4)
            # Current and Target Displacement (using palm displacement)
            # Actions (20)
            # Total: 44
            obs_idx = 0
            # Fingertip position
            self.obs_buf[:, obs_idx:obs_idx+15] = self.fingertip_pos.reshape(self.num_envs, 15)
            obs_idx += 15
            # Rope position
            self.obs_buf[:, obs_idx:obs_idx+3] = self.rope_pos
            obs_idx += 3
            # Rope rotation
            self.obs_buf[:, obs_idx:obs_idx+4] = self.rope_rot
            obs_idx += 4
            # Current and Target Displacement (using palm displacement)
            self.obs_buf[:, obs_idx] = current_displacement
            obs_idx += 1
            # Ensure local_rope_end_offset has correct shape (N, 1) -> (N,)
            self.obs_buf[:, obs_idx] = self.target_displacement
            obs_idx += 1
            # Actions
            self.obs_buf[:, obs_idx:obs_idx+self.num_actions] = self.actions
            obs_idx += self.num_actions
        else:
             # Needs significant rework to include rope state + velocities
            print("Warning: 'openai' observation type with velocities not fully implemented for rope task.")
            pass # Placeholder

    def compute_full_observations(self, no_vel=False):
        # Placeholder - Needs modification for rope task
        # Use pre-calculated palm displacement
        current_displacement = self.current_palm_displacement

        # Structure:
        # DOF pos (24) | DOF vel (24, if not no_vel) | Rope Pose (7) | Rope Vel (6, if not no_vel)
        # Fingertip State (15 or 65) | Current Disp (1) | Target Disp (1) | Actions (20)

        obs_idx = 0
        # DOF Pos
        self.obs_buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        obs_idx += self.num_shadow_hand_dofs

        if not no_vel:
            # DOF Vel
            self.obs_buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            obs_idx += self.num_shadow_hand_dofs

        # Rope Pose
        self.obs_buf[:, obs_idx:obs_idx+7] = self.rope_pose
        obs_idx += 7

        if not no_vel:
            # Rope Vel
            self.obs_buf[:, obs_idx:obs_idx+3] = self.rope_linvel
            obs_idx += 3
            self.obs_buf[:, obs_idx:obs_idx+3] = self.vel_obs_scale * self.rope_angvel
            obs_idx += 3

        # Fingertip State (position only or full state)
        if no_vel:
            num_ft_dims = 15
            ft_data = self.fingertip_pos.reshape(self.num_envs, num_ft_dims)
        else:
            num_ft_dims = 65
            ft_data = self.fingertip_state.reshape(self.num_envs, num_ft_dims)
        self.obs_buf[:, obs_idx:obs_idx+num_ft_dims] = ft_data
        obs_idx += num_ft_dims

        # Current and Target Displacement (using palm displacement)
        self.obs_buf[:, obs_idx] = current_displacement
        obs_idx += 1
        # Ensure local_rope_end_offset has correct shape (N, 1) -> (N,)
        self.obs_buf[:, obs_idx] = self.target_displacement
        obs_idx += 1

        # Actions
        self.obs_buf[:, obs_idx:obs_idx+self.num_actions] = self.actions
        obs_idx += self.num_actions

        # Zero out remaining buffer if needed (optional, ensure numObservations is accurate)
        # if obs_idx < self.num_observations:
        #     self.obs_buf[:, obs_idx:] = 0

    def compute_full_state(self, asymm_obs=False):
        # Placeholder - Needs modification for rope task
        # Use pre-calculated palm displacement
        current_displacement = self.current_palm_displacement

        # Structure:
        # DOF pos (24) | DOF vel (24) | DOF force (24) | Rope Pose (7) | Rope Vel (6)
        # Fingertip Full State (65) | Fingertip Forces (30) |
        # Current Disp (1) | Target Disp (1) | Actions (20)

        buf = self.states_buf if asymm_obs else self.obs_buf
        obs_idx = 0
        # Hand state (pos, vel, force)
        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        obs_idx += self.num_shadow_hand_dofs
        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        obs_idx += self.num_shadow_hand_dofs
        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor
        obs_idx += self.num_shadow_hand_dofs

        # Rope state (pose, vel)
        buf[:, obs_idx:obs_idx+7] = self.rope_pose
        obs_idx += 7
        buf[:, obs_idx:obs_idx+3] = self.rope_linvel
        obs_idx += 3
        buf[:, obs_idx:obs_idx+3] = self.vel_obs_scale * self.rope_angvel
        obs_idx += 3

        # Fingertip state (full state + sensors)
        num_ft_states = 13 * self.num_fingertips # 65
        num_ft_force_torques = 6 * self.num_fingertips # 30
        buf[:, obs_idx:obs_idx + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        obs_idx += num_ft_states
        buf[:, obs_idx : obs_idx + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor
        obs_idx += num_ft_force_torques

        # Current and Target Displacement (using palm displacement)
        buf[:, obs_idx] = current_displacement
        obs_idx += 1
        # Ensure local_rope_end_offset has correct shape (N, 1) -> (N,)
        buf[:, obs_idx] = self.target_displacement
        obs_idx += 1

        # Actions
        buf[:, obs_idx:obs_idx + self.num_actions] = self.actions
        obs_idx += self.num_actions

        # Zero out remaining buffer if needed (optional, ensure numObservations/numStates is accurate)
        # num_total_dims = self.num_states if asymm_obs else self.num_observations
        # if obs_idx < num_total_dims:
        #      buf[:, obs_idx:] = 0
    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Updates target poses for specified environments
        """
        # Get current target displacements
        current_targets = self.target_displacement[env_ids]

        # Generate new target displacements (alternating between positive and negative)
        new_targets = -current_targets  # Simple flip between positive and negative

        # Update targets
        self.target_displacement[env_ids] = new_targets

        if apply_reset:
            # Reset rope positions if needed
            self.root_state_tensor[self.rope_indices[env_ids]] = self.rope_init_state[env_ids].clone()
            rope_indices_int32 = self.rope_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self.root_state_tensor),
                                                       gymtorch.unwrap_tensor(rope_indices_int32),
                                                       len(rope_indices_int32))
        # <<< Add clearing of reset_goal_buf >>>
        if len(env_ids) > 0: # Check to prevent error if env_ids is empty
            self.reset_goal_buf[env_ids] = 0
        # <<< End clearing of reset_goal_buf >>>

    def reset_idx(self, env_ids):
        # Resetting logic needs careful adaptation for the rope task
        # This version mostly resets hand and rope to initial state with noise

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # <<< Call reset_target_pose within reset_idx >>>
        # This ensures target is flipped/updated on full environment resets, similar to Spin
        if len(env_ids) > 0: # Ensure env_ids is not empty before calling
            self.reset_target_pose(env_ids)
        # <<< End call reset_target_pose >>>

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # Reset rope - add noise to initial pose
        # *** Use rope_init_state ***
        self.root_state_tensor[self.rope_indices[env_ids]] = self.rope_init_state[env_ids].clone()
        # Add position noise (example, adjust as needed)
        self.root_state_tensor[self.rope_indices[env_ids], 0:3] += self.reset_position_noise * rand_floats[:, 0:3]
        # Add rotation noise (example, adjust as needed)
        # new_rope_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        # self.root_state_tensor[self.rope_indices[env_ids], 3:7] = new_rope_rot # Apply if needed
        self.root_state_tensor[self.rope_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.rope_indices[env_ids], 7:13])

        # Set rope root state
        rope_indices_int32 = self.rope_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(rope_indices_int32), len(rope_indices_int32))

        # reset random force probabilities for the rope
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_shadow_hand_dofs] + 1)

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        # Set hand state
        hand_indices_int32 = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices_int32), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices_int32), len(env_ids))

        # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        # <<< Start: Reset target displacement to initial positive value >>>
        # self.local_rope_end_offset[env_ids] = self.target_displacement_positive_val # REMOVED - target handling moved
        # <<< End: Reset target displacement to initial positive value >>>

    def pre_physics_step(self, actions):
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1) # OLD LOGIC

        # # Reset environments that need it # OLD LOGIC
        # if len(env_ids) > 0: # OLD LOGIC
        #     self.reset_idx(env_ids) # OLD LOGIC

        # <<< Start: New pre_physics_step logic based on ShadowHandSpin >>>
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # Handle goal resets (target flip).
        # In Spin, reset_target_pose clears the reset_goal_buf.
        if len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids) # apply_reset not directly applicable / needed for rope

        # Handle full environment resets.
        # reset_idx will also call reset_target_pose for these env_ids.
        # If an env is in both goal_env_ids and env_ids_to_reset, its target will be flipped twice
        # (once by the call above, once by the call inside reset_idx). This matches Spin's behavior.
        if len(env_ids_to_reset) > 0:
            self.reset_idx(env_ids_to_reset)
        # <<< End: New pre_physics_step logic >>>

        # Apply actions
        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # Apply moving average if configured
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices] + \
                                                               (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # Apply random forces if configured
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            # Apply forces only to rope rigid bodies
            # *** Use rope_rb_handles and rope_rb_masses ***
            self.rb_forces[force_indices, self.rope_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.rope_rb_handles, :].shape, device=self.device) * self.rope_rb_masses * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        if self.randomize:
            self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions) # Compute placeholder reward

        # Debug visualization (optional)
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # Add visualization for rope if needed
            # ... similar to goal visualization in Spin task ...

    


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def compute_hand_reward( 
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes, # Added reset_goal_buf
    max_episode_length: float, current_displacement, target_displacement,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float, # rot params added for signature match, may not be used
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float
):
    # Calculate displacement error and reward
    displacement_error = torch.abs(current_displacement - target_displacement)
    # Using exponential reward based on original class method
    dist_reward = torch.exp(-dist_reward_scale * displacement_error)

    # Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + action regularization + success bonus + fall penalty
    # Using user's formula: dist_reward + action_penalty * scale
    # Original class method included fall_penalty directly here.
    # User JIT calculates fall penalty separately and applies later? No, seems it uses fall_dist for resets.
    # Let's stick to user's JIT structure:
    reward = dist_reward - action_penalty * action_penalty_scale # Changed '+' to '-' assuming penalty

    # Find out which envs hit the goal and update successes count
    # Comparing error to tolerance
    # <<< This becomes reset_goal_buf_out >>>
    reset_goal_buf_out = torch.where(displacement_error <= success_tolerance, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf)) # Use zeros_like for initial state if not goal
    # Increment successes only if reset is 1 (i.e., goal reached)
    successes = successes + reset_goal_buf_out # Use reset_goal_buf_out here

    # Success bonus: displacement is within `success_tolerance` of target
    reward = torch.where(reset_goal_buf_out == 1, reward + reach_goal_bonus, reward)

    # Determine resets based on fall distance (displacement error too large)
    fall_resets = torch.where(displacement_error >= fall_dist, torch.ones_like(reset_buf), torch.zeros_like(reset_buf))
    # Apply fall penalty where fall_resets is 1
    reward = torch.where(fall_resets == 1, reward + fall_penalty, reward)

    # Combine resets: reset if falling OR max episode length reached OR max consecutive successes reached
    resets = fall_resets

    # Handle max consecutive successes reset logic
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        # Important: Only reset progress if goal was JUST reached (reset_goal_buf_out == 1)
        progress_buf = torch.where(reset_goal_buf_out == 1, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets) # Reset if max successes reached

    # Handle max episode length reset
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for timing out (not reaching goal by max episode length)
    # This seems different from user's JIT, adding based on original shadow_hand logic
    # reward = torch.where((progress_buf >= max_episode_length - 1) & (reset_goal_buf_out == 0), reward + 0.5 * fall_penalty, reward) # Penalize timeout only if not successful

    # Calculate average consecutive successes
    # Need to ensure successes are reset to 0 when env resets
    # Note: This calculation happens *before* successes are potentially reset below
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float()) # Sum successes for envs that are resetting *now*

    # Reset successes count for envs that are resetting
    successes = torch.where(resets == 1, torch.zeros_like(successes), successes)

    # Calculate the running average
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    # Reset progress buf for envs that reset (already handled for goal resets if max_consecutive > 0)
    # Ensure progress buf is reset for fall resets and max length resets too
    progress_buf = torch.where(resets == 1, torch.zeros_like(progress_buf), progress_buf)

    # The user's JIT function signature implies returning goal_resets is not needed,
    # unlike the original JIT function. We adapt to return what the class method expects.
    # The class method now expects: reward, resets, progress_buf, successes, cons_successes
    # <<< Update JIT return >>>
    return reward, resets, reset_goal_buf_out, progress_buf, successes, cons_successes
# <<< End: Add user-provided JIT function >>> 
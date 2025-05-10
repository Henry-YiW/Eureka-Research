class ShadowHandRope(VecTask):
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
        buf[:, obs_idx] = self.local_rope_end_offset.squeeze(-1)
        obs_idx += 1

        # Actions
        buf[:, obs_idx:obs_idx + self.num_actions] = self.actions
        obs_idx += self.num_actions

        # Zero out remaining buffer if needed (optional, ensure numObservations/numStates is accurate)
        # num_total_dims = self.num_states if asymm_obs else self.num_observations
        # if obs_idx < num_total_dims:
        #      buf[:, obs_idx:] = 0

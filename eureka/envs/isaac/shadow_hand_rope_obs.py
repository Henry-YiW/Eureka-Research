class ShadowHandRope(VecTask):
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.rope_pose = self.root_state_tensor[self.rope_indices, 0:7]
        self.rope_pos = self.root_state_tensor[self.rope_indices, 0:3]
        self.rope_rot = self.root_state_tensor[self.rope_indices, 3:7]
        self.rope_linvel = self.root_state_tensor[self.rope_indices, 7:10]
        self.rope_angvel = self.root_state_tensor[self.rope_indices, 10:13]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        palm_pos = self.rigid_body_states[:, self.palm_handle, 0:3]
        rope_y_axis_world = quat_apply(self.rope_rot, self.local_y_axis)
        vec_rope_to_palm = palm_pos - self.rope_pos
        self.current_palm_displacement = torch.sum(vec_rope_to_palm * rope_y_axis_world, dim=1)

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

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        buf = self.states_buf if asymm_obs else self.obs_buf
        obs_idx = 0

        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = unscale(
            self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        obs_idx += self.num_shadow_hand_dofs
        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        obs_idx += self.num_shadow_hand_dofs
        buf[:, obs_idx:obs_idx+self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor
        obs_idx += self.num_shadow_hand_dofs

        buf[:, obs_idx:obs_idx+7] = self.rope_pose
        obs_idx += 7
        buf[:, obs_idx:obs_idx+3] = self.rope_linvel
        obs_idx += 3
        buf[:, obs_idx:obs_idx+3] = self.vel_obs_scale * self.rope_angvel
        obs_idx += 3

        num_ft_states = 13 * self.num_fingertips
        num_ft_force_torques = 6 * self.num_fingertips

        buf[:, obs_idx:obs_idx+num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        obs_idx += num_ft_states
        buf[:, obs_idx:obs_idx+num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor
        obs_idx += num_ft_force_torques

        buf[:, obs_idx] = self.current_palm_displacement
        obs_idx += 1
        buf[:, obs_idx] = self.local_rope_end_offset.squeeze(-1)
        obs_idx += 1

        buf[:, obs_idx:obs_idx+self.num_actions] = self.actions

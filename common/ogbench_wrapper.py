"""OGBench environment wrapper: torch I/O, the supervised goal reward for the replay buffer, and evaluation rendering."""

from collections import defaultdict
import numpy as np
import torch

import gymnasium as gym
import ogbench
import mujoco

from common.utils.misc import (
	SUBGOAL_AGENTS,
	plot_state_value_on_frame,
	plot_value_on_maze,
	plot_plan_on_maze,
	plot_state_xy_on_maze,
	plot_states_xy_on_maze,
)

# OGBench layouts. qpos: cube i is qpos[14 + 7i : 21 + 7i] (3 pos + 4 quat); antsoccer ball xy is qpos[-7:-5].
# manipulation state obs: end-effector xyz at [12:15], cube i's 9-d block at [19 + 9i : 28 + 9i] (xyz first);
# antsoccer state obs: ball xy at [15:17].
QPOS_CUBE_START, QPOS_CUBE_LEN = 14, 7
QPOS_BALL_XY = slice(-7, -5)
OBS_EE_XYZ = slice(12, 15)
OBS_CUBE_START, OBS_CUBE_LEN = 19, 9
OBS_BALL_XY = slice(15, 17)


class OGBenchWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	Also handles reward, rendering and visualization.
	"""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.cfg = cfg
		if 'maze' in cfg.task or 'soccer' in cfg.task:
			self.qmap = None  # for visualization purposes
		if 'soccer' in cfg.task:
			self._last_value_map_ball_xy = None
			self._ball_xy_eps = 1e-3
		# obs=latent: frames are encoded with the frozen autoencoder (render() still returns pixels)
		self.ae = None
		if cfg.obs == 'latent':
			from agents.modules.autoencoder import Autoencoder
			self.ae = Autoencoder.from_checkpoint(cfg.latent_encoder_checkpoint, device='cuda:0')
			print(f"Encoding observations with the autoencoder from {cfg.latent_encoder_checkpoint} (z_dim={self.ae.z_dim})")

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		return self._encode(self._try_f32_tensor(obs))

	@torch.no_grad()
	def _encode(self, obs):
		"""[H, W, C] uint8 frame -> [z_dim] latent under obs=latent, else a no-op."""
		if self.ae is None or obs is None:
			return obs
		z = self.ae.encode(obs.unsqueeze(0).to('cuda:0'))
		return z.squeeze(0).cpu()

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		# convert goal to tensor
		goal = info.get('goal')
		goal = self._obs_to_tensor(goal)
		info["goal"] = goal
		# convert observation to tensor
		obs = self._obs_to_tensor(obs)
		if 'soccer' in self.cfg.task:
			self.qmap = None
			self._last_value_map_ball_xy = None
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		reward = 0.0 if info['success'] else -1.0
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), terminated, truncated, info
	
	def compute_reward_batch(self, qpos, goal_qpos, button_states, goal_button_states):
		"""Compute the supervised reward for a batch of observations and goals from buffer."""
		
		if "maze" in self.cfg.task:
			goal_tol = 1.0 if 'point' in self.cfg.task else 0.5
			goal_xy = goal_qpos[:, :2]
			agent_xy = qpos[:, :2]
			dists = torch.linalg.norm(goal_xy - agent_xy, dim=-1)  # (batch * horizon,)
			reward = torch.where(dists <= goal_tol, 0.0, -1.0)
		
		elif "soccer" in self.cfg.task:
			ball_xy = qpos[:, QPOS_BALL_XY]
			goal_ball_xy = goal_qpos[:, QPOS_BALL_XY]
			dists = torch.linalg.norm(goal_ball_xy - ball_xy, dim=-1)  # (batch * horizon,)
			reward = torch.where(dists <= 0.5, 0.0, -1.0)
		
		elif "cube" in self.cfg.task:
			# success: every cube within 0.04 of its target position
			num_cubes = self.env.unwrapped._num_cubes
			cube_xyzs = self._cube_xyzs(qpos, num_cubes)
			goal_xyzs = self._cube_xyzs(goal_qpos, num_cubes)
			all_success = torch.linalg.norm(goal_xyzs - cube_xyzs, dim=-1) <= 0.04  # (batch * horizon, num_cubes)
			reward = torch.where(torch.all(all_success, dim=-1, keepdim=True), 0.0, -1.0)

		elif "scene" in self.cfg.task:
			num_cubes = self.env.unwrapped._num_cubes
			num_buttons = self.env.unwrapped._num_buttons
			qpos_drawer_idx = QPOS_CUBE_START + num_cubes * QPOS_CUBE_LEN + num_buttons
			qpos_window_idx = qpos_drawer_idx + 1

			cube_xyzs = self._cube_xyzs(qpos, num_cubes)
			goal_xyzs = self._cube_xyzs(goal_qpos, num_cubes)
			cube_successes = torch.linalg.norm(goal_xyzs - cube_xyzs, dim=-1) <= 0.04
			button_successes = button_states == goal_button_states
			drawer_success = torch.abs(qpos[:, qpos_drawer_idx] - goal_qpos[:, qpos_drawer_idx]) <= 0.04
			window_success = torch.abs(qpos[:, qpos_window_idx] - goal_qpos[:, qpos_window_idx]) <= 0.04
			successes = torch.cat([cube_successes, button_successes, drawer_success[:, None], window_success[:, None]], dim=-1)
			
			reward = successes.all(dim=-1, keepdim=True).float() - 1
			
		else:
			raise NotImplementedError(f"Task reward for '{self.cfg.task}' is not implemented.")

		return reward

	@staticmethod
	def _cube_xyzs(qpos, num_cubes):
		"""(N, nq) qpos -> (N, num_cubes, 3) cube positions."""
		cubes = qpos[:, QPOS_CUBE_START : QPOS_CUBE_START + num_cubes * QPOS_CUBE_LEN]
		return cubes.reshape(-1, num_cubes, QPOS_CUBE_LEN)[..., :3]

	@staticmethod
	def _unnormalize(x, info):
		"""Normalized observations -> raw observation space, with the dataset statistics passed in info."""
		return x * info["obs_std"].to(x.device, x.dtype) + info["obs_mean"].to(x.device, x.dtype)

	def _add_sphere_marker(self, scene, pos, radius, rgba):
		if scene.ngeom >= scene.maxgeom:
			return
		geom = scene.geoms[scene.ngeom]
		size = np.array([radius, radius, radius], dtype=np.float64)
		mat = np.eye(3, dtype=np.float64).reshape(-1)
		mujoco.mjv_initGeom(
			geom,
			mujoco.mjtGeom.mjGEOM_SPHERE,
			size,
			np.asarray(pos, dtype=np.float64),
			mat,
			np.asarray(rgba, dtype=np.float32),
		)
		# decor geoms are skipped by MuJoCo's shadow pass, so markers cast no shadows
		geom.category = mujoco.mjtCatBit.mjCAT_DECOR
		scene.ngeom += 1

	def render(self, info=None):
		info = {} if info is None else info

		# image / latent observations have no world coordinates to overlay
		if self.cfg.obs in ('rgb', 'latent'):
			return super().render().copy()

		if 'maze' in self.cfg.task:
			frame = super().render().copy()
			frame, self.qmap = plot_value_on_maze(frame, self.qmap, info)
			maze_type = info["maze_type"]
			H, W = frame.shape[:2]
			# marker radii are tuned for 200x200 frames
			px_scale = max(1, round(min(W, H) / 200))
			if self.cfg.agent in SUBGOAL_AGENTS:
				subgoal = info["subgoal"]
				subgoal_samples = info.get("subgoal_samples")
				if subgoal_samples is not None:
					unnormalized_subgoal_samples = self._unnormalize(subgoal_samples, info)
					frame = plot_states_xy_on_maze(frame, unnormalized_subgoal_samples, color=(255, 255, 255), maze_type=maze_type, alpha=0.25, radius=2 * px_scale)
				unnormalized_subgoal = self._unnormalize(subgoal, info)
				frame = plot_state_xy_on_maze(frame, unnormalized_subgoal, color=(0, 255, 0), maze_type=maze_type, width=W, height=H, alpha=1.0, radius=3 * px_scale)
			if self.cfg.agent == 'tdmpc':
				frame = plot_plan_on_maze(frame, info)

		elif 'soccer' in self.cfg.task:
			frame = super().render().copy()
			if info.get("value_fn") is not None:
				# Heatmap is V(ant_xy | ball, goal); recompute only when ball xy changes.
				ball_xy = info["obs"].detach().float().reshape(-1)[OBS_BALL_XY].cpu().numpy()
				if (
					self.qmap is None
					or self._last_value_map_ball_xy is None
					or np.linalg.norm(ball_xy - self._last_value_map_ball_xy) > self._ball_xy_eps
				):
					info["compute_value_map"] = True
					self._last_value_map_ball_xy = ball_xy.copy()
				else:
					info["compute_value_map"] = False
				frame, self.qmap = plot_value_on_maze(frame, self.qmap, info)

			# Subgoal overlays: ant xy (reuse maze colors) + ball xy (distinct colors).
			# Selected subgoals are drawn larger with a dark outline for emphasis.
			if self.cfg.agent in SUBGOAL_AGENTS and info.get("subgoal") is not None:
				maze_type = info["maze_type"]
				H, W = frame.shape[:2]
				# marker radii are tuned for 200x200 frames
				px_scale = max(1, round(min(W, H) / 200))
				subgoal = info["subgoal"]
				subgoal_samples = info.get("subgoal_samples")
				ant_xy_slice = slice(0, 2)
				ball_xy_slice = OBS_BALL_XY
				def _plot_selected(frame, unnorm_sg, color, xy_slice):
					# Dark outline then filled marker so the chosen subgoal stands out.
					frame = plot_state_xy_on_maze(
						frame, unnorm_sg, color=(0, 0, 0), maze_type=maze_type,
						width=W, height=H, alpha=1.0, radius=3 * px_scale, xy_slice=xy_slice,
					)
					frame = plot_state_xy_on_maze(
						frame, unnorm_sg, color=color, maze_type=maze_type,
						width=W, height=H, alpha=1.0, radius=2 * px_scale, xy_slice=xy_slice,
					)
					return frame

				if subgoal_samples is not None:
					unnormalized_subgoal_samples = self._unnormalize(subgoal_samples, info)
					frame = plot_states_xy_on_maze(
						frame, unnormalized_subgoal_samples, color=(255, 255, 255), maze_type=maze_type,
						alpha=0.25, xy_slice=ant_xy_slice, radius=2 * px_scale,
					)
					frame = plot_states_xy_on_maze(
						frame, unnormalized_subgoal_samples, color=(160, 32, 240), maze_type=maze_type,
						alpha=0.25, xy_slice=ball_xy_slice, radius=2 * px_scale,
					)
				unnormalized_subgoal = self._unnormalize(subgoal, info)
				frame = _plot_selected(frame, unnormalized_subgoal, (0, 255, 0), ant_xy_slice)
				frame = _plot_selected(frame, unnormalized_subgoal, (160, 32, 240), ball_xy_slice)
	
		elif "cube" in self.cfg.task and self.cfg.agent in SUBGOAL_AGENTS:
			# OGBench CubeEnv state obs stores (world_xyz - center) * scaler for the end-effector (OBS_EE_XYZ) and each
			# cube (first 3 entries of its OBS_CUBE_LEN block), so world xyz = scaled / scaler + center.
			sg = info["subgoal"].reshape(-1)  # [obs_dim], columns match CubeEnv state observation (not only xyz)
			unnorm_subgoal_obs = self._unnormalize(sg, info).detach().cpu().numpy().astype(np.float64)

			center = np.array([0.425, 0.0, 0.0], dtype=np.float64)  # matches OGBench CubeEnv.compute_observation constant
			scaler = 10.0  # matches OGBench CubeEnv.compute_observation constant
			num_cubes = int(self.env.unwrapped._num_cubes)

			ee_world = unnorm_subgoal_obs[OBS_EE_XYZ] / scaler + center
			cube_cols = OBS_CUBE_START + np.arange(num_cubes)[:, None] * OBS_CUBE_LEN + np.arange(3)
			cubes_world = unnorm_subgoal_obs[cube_cols] / scaler + center  # (num_cubes, 3)

			ogb = getattr(self.env.unwrapped, "_colors", {})
			cube_rgb = []
			for key in ("red", "blue", "orange", "green", "purple", "yellow", "magenta", "gray"):
				c = ogb.get(key, np.ones(4, dtype=np.float64))
				cube_rgb.append([float(c[0]), float(c[1]), float(c[2])])
			cube_rgb = np.asarray(cube_rgb, dtype=np.float64)
			p = ogb.get("purple", np.array([0.61, 0.28, 0.82, 1.0], dtype=np.float64))
			ee_rgb = (float(p[0]), float(p[1]), float(p[2]))
			n_c = cube_rgb.shape[0]

			r_obj = 0.014
			r_chosen = r_obj

			spheres = []

			# Optional: visualize subgoal candidates (semi-transparent) behind chosen spheres.
			if getattr(self.cfg, "render_subgoal_candidates", False) and "subgoal_samples" in info:
				sg_samples = info["subgoal_samples"].reshape(-1, sg.shape[-1])  # [n_sg_samples, obs_dim]
				unnorm_subgoal_samples = self._unnormalize(sg_samples, info).detach().cpu().numpy().astype(np.float64)

				alpha_candidates = 0.15
				r_cand = r_obj * 0.4
				for cand in unnorm_subgoal_samples:
					cubes_world_cand = cand[cube_cols] / scaler + center  # (num_cubes, 3)
					ee_world_cand = cand[OBS_EE_XYZ] / scaler + center  # (3,)
					for i in range(num_cubes):
						rgb = cube_rgb[i % n_c]
						spheres.append((cubes_world_cand[i], r_cand, (rgb[0], rgb[1], rgb[2], alpha_candidates)))
					spheres.append((ee_world_cand, r_cand, (*ee_rgb, alpha_candidates)))

			for i in range(num_cubes):
				rgb = cube_rgb[i % n_c]
				spheres.append((cubes_world[i], r_chosen, (rgb[0], rgb[1], rgb[2], 1.0)))
			spheres.append((ee_world, r_chosen, (*ee_rgb, 1.0)))

			def _draw_subgoal_spheres(scene):
				for pos, rad, rgba in spheres:
					self._add_sphere_marker(scene, pos, radius=rad, rgba=rgba)

			frame = self.env.unwrapped.render(scene_callback=_draw_subgoal_spheres).copy()
			frame = plot_state_value_on_frame(frame, info)

		elif "scene" in self.cfg.task and self.cfg.agent in SUBGOAL_AGENTS:
			# SceneEnv.compute_observation constants (see ogbench/manipspace/envs/scene_env.py):
			# - xyz_center=[0.425,0,0], xyz_scaler=10
			# - buttons: one-hot(2) + button_joint_pos*120 + button_joint_vel (per button)
			# - drawer_pos*18, window_pos*15 (plus velocities)
			sg = info["subgoal"].reshape(-1)  # [obs_dim]
			unnorm_subgoal_obs = self._unnormalize(sg, info).detach().cpu().numpy().astype(np.float64)

			center = np.array([0.425, 0.0, 0.0], dtype=np.float64)
			xyz_scaler = 10.0
			button_scaler = 120.0
			drawer_scaler = 18.0
			window_scaler = 15.0

			env = self.env.unwrapped
			num_cubes = int(getattr(env, "_num_cubes", 1))
			num_buttons = int(getattr(env, "_num_buttons", 2))

			# Extract world-space cube position subgoals.
			cube_cols = OBS_CUBE_START + np.arange(num_cubes)[:, None] * OBS_CUBE_LEN + np.arange(3)
			cubes_world = unnorm_subgoal_obs[cube_cols] / xyz_scaler + center  # (num_cubes, 3)

			# Extract button one-hot states from subgoals.
			btn0 = OBS_CUBE_START + num_cubes * OBS_CUBE_LEN
			btn_stride = 2 + 1 + 1  # one-hot(2) + pos(1) + vel(1)
			button_onehots = []
			for i in range(num_buttons):
				start = btn0 + i * btn_stride
				button_onehots.append(unnorm_subgoal_obs[start : start + 2])
			button_onehots = np.stack(button_onehots, axis=0)  # (num_buttons, 2)

			# Drawer/window scalar subgoals (unnormalize scaling only).
			after_buttons = btn0 + num_buttons * btn_stride
			drawer_pos = float(unnorm_subgoal_obs[after_buttons + 0] / drawer_scaler)
			window_pos = float(unnorm_subgoal_obs[after_buttons + 2] / window_scaler)

			# Drawer/window subgoal scalars are absolute slide joint qpos (same quantities SceneEnv uses vs targets).
			# Convert qpos deltas -> world xyz using the empirically-correct slide axis derived from
			# current handle vs the environment's *target* handle.
			#
			# This avoids assuming the drawer/window translation is aligned to world axes.
			drawer_handle = env.data.site_xpos[env._drawer_site_id].copy()
			window_handle = env.data.site_xpos[env._window_site_id].copy()
			q_drawer_now = float(env.data.joint("drawer_slide").qpos[0])
			q_window_now = float(env.data.joint("window_slide").qpos[0])

			q_drawer_goal = float(getattr(env, "_target_drawer_pos", q_drawer_now))
			q_window_goal = float(getattr(env, "_target_window_pos", q_window_now))
			drawer_target = env.data.site_xpos[env._drawer_target_site_id].copy()
			window_target = env.data.site_xpos[env._window_target_site_id].copy()

			dq_drawer = q_drawer_goal - q_drawer_now
			dq_window = q_window_goal - q_window_now

			# Axis in world coordinates, scaled as (world_delta_xyz_per_unit_joint).
			if abs(dq_drawer) > 1e-6:
				drawer_axis = (drawer_target - drawer_handle) / dq_drawer
			else:
				# Degenerate case: current ~= goal; fall back to world-Y assumption.
				drawer_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

			if abs(dq_window) > 1e-6:
				window_axis = (window_target - window_handle) / dq_window
			else:
				# Degenerate case: current ~= goal; fall back to world-X assumption.
				window_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

			# Button state bulbs: fixed world-frame offsets from each button top site (SceneEnv — no camera queries).
			# Tuned so markers sit slightly in front of the row of buttons in the default “front” view.
			scene_button_bulb_offsets = (
				np.array([0.10, -0.035, 0.065], dtype=np.float64),
				np.array([0.10, 0.035, 0.065], dtype=np.float64),
			)
			button_bulbs = []
			for i in range(num_buttons):
				btn_pos = env.data.site_xpos[env._button_site_ids[i]].copy()
				d = scene_button_bulb_offsets[i % len(scene_button_bulb_offsets)]
				button_bulbs.append(btn_pos + d)

			# Colors: cube red; drawer/window blue; button bulbs white (unlocked) or red (locked).
			ogb = getattr(env, "_colors", {})
			red = ogb.get("red", np.array([0.96, 0.26, 0.33, 1.0], dtype=np.float64))
			blue = ogb.get("blue", np.array([0.35, 0.55, 0.91, 1.0], dtype=np.float64))
			white = ogb.get("white", np.array([0.9, 0.9, 0.9, 1.0], dtype=np.float64))
			r_rgb = (float(red[0]), float(red[1]), float(red[2]))
			b_rgb = (float(blue[0]), float(blue[1]), float(blue[2]))
			w_rgb = (float(white[0]), float(white[1]), float(white[2]))

			alpha = 0.5
			rad = 0.014
			spheres = []
			# Cube (always red)
			spheres.append((cubes_world[0], rad, (*r_rgb, alpha)))
			# Drawer + window (blue)
			drawer_xyz = drawer_handle + drawer_axis * (drawer_pos - q_drawer_now)
			window_xyz = window_handle + window_axis * (window_pos - q_window_now)
			spheres.append((drawer_xyz, rad, (*b_rgb, alpha)))
			spheres.append((window_xyz, rad, (*b_rgb, alpha)))
			# Buttons: bulb position fixed; color depends on subgoal one-hot (state 0=locked/red, state 1=unlocked/white)
			for i in range(num_buttons):
				oh = button_onehots[i]
				locked = float(oh[0]) >= float(oh[1])
				rgb = r_rgb if locked else w_rgb
				spheres.append((button_bulbs[i], rad * 0.85, (*rgb, alpha)))

			def _draw_scene_subgoal_spheres(scene):
				for pos, rad, rgba in spheres:
					self._add_sphere_marker(scene, pos, radius=rad, rgba=rgba)

			frame = env.render(scene_callback=_draw_scene_subgoal_spheres).copy()
			frame = plot_state_value_on_frame(frame, info)
		
		else:
			frame = super().render().copy()

		return frame


def make_env(cfg):
	"""
	Make the wrapped OGBench evaluation environment for cfg.task, and set the environment-derived cfg fields
	(env_episode_length, obs_shape, action_dim, action_high, action_low).
	"""
	env = ogbench.make_env_and_datasets(cfg.task, env_only=True)
	if 'visual' in cfg.task:
		assert cfg.obs in ['rgb', 'latent'], f"visual tasks require image observations (got obs={cfg.obs})"

	if "large" in cfg.task or "giant" in cfg.task:
		env._max_episode_steps = 2000
	elif "antsoccer" in cfg.task:
		env._max_episode_steps = 2000
	elif "double" in cfg.task or "triple" in cfg.task:
		env._max_episode_steps = 1000
	cfg.env_episode_length = env._max_episode_steps

	env = OGBenchWrapper(env, cfg)

	if cfg.obs == 'latent':
		cfg.obs_shape = (env.ae.z_dim,)
	else:
		cfg.obs_shape = env.observation_space.shape
	cfg.action_dim = env.action_space.shape[0]
	cfg.action_high = env.action_space.high[0].item()
	cfg.action_low = env.action_space.low[0].item()

	return env

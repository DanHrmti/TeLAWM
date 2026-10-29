import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from common.utils.rms_norm import RMSNormalizer


class _EpisodeTensor:
	"""Lazy [episode, timestep] view over a flat storage tensor: gathers only the indexed frames.
	`base` holds the flat index of each sampled episode's first step."""

	def __init__(self, flat, base):
		self._flat = flat
		self._base = base

	def __getitem__(self, idx):
		ep_idx, time_idx = idx
		return self._flat[self._base[ep_idx] + time_idx]


class _EpisodeBatch:
	"""Lazy stand-in for a batch of sampled episodes: returns an `_EpisodeTensor` per key."""

	def __init__(self, data, base):
		self._data = data
		self._base = base

	def get(self, key):
		return _EpisodeTensor(self._data[key], self._base)


class Buffer():
	"""
	Replay buffer for training, based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg, env, capacity):
		self.cfg = cfg
		self.env = env
		self._device = torch.device('cuda:0')
		self._capacity = capacity
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=True, # NOTE: should be True in offline training
			compile=cfg.compile,
		)
		self.normalizer = RMSNormalizer(self.cfg.obs_shape[-1], disable=(self.cfg.obs=='rgb'))  # initialize RMS normalizer
		# lazy gather: index the stored dataset directly instead of materializing full episodes (default: on for rgb)
		lazy = self.cfg.get("buffer_lazy_gather", "auto")
		self._lazy = (self.cfg.obs == 'rgb') if lazy in ("auto", None) else bool(lazy)

		self._batch_size = cfg.batch_size * cfg.data_episode_length
		self._num_eps = 0
		if self.cfg.geom_sample_value or self.cfg.geom_sample_policy:
			self._geometric_dist = torch.distributions.Geometric(torch.tensor([1 - self.cfg.gamma], device=self._device))
		# IQL n-step TD: sample a length-n trajectory window and aggregate into a 1-step batch
		self.horizon = int(self.cfg.iql['n_step']) if self.cfg.agent == "iql" else 1

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps
	
	@property
	def obs_mean(self):
		"""Mean of the observation normalizer."""
		return self.normalizer.mean
	
	@property
	def obs_std(self):
		"""Standard deviation of the observation normalizer."""
		return torch.sqrt(self.normalizer.var)

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, td):
		"""Initialize the replay buffer. Use the first tensordict to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in td.values()
		]) / len(td)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		if self._lazy:
			storage_device = 'cuda:0' if 1.15*total_bytes + 3e9 < mem_free else 'cpu'
		else:
			storage_device = 'cuda:0' if 2.0*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		if self._lazy:
			return None
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td) // (self.cfg.data_episode_length + 1) 
		td['episode'] = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64).repeat_interleave(self.cfg.data_episode_length + 1)
		if self._num_eps == 0:
			self._buffer = self._init(td[0].unsqueeze(0))
			self.normalizer.set_stats(td["obs"])  # set RMS normalizer stats once from offline data
		if self._lazy:
			assert self._num_eps == 0, "the lazy-gather buffer is loaded once (offline training)"
			self._data = td.to(self._storage_device)
		else:
			self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps
	
	def add(self, td):
		"""Add an episode to the buffer."""
		assert not self._lazy, "the lazy-gather buffer is offline only (use load())"
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self.normalizer.update(td["obs"])  # update RMS normalizer stats
		self._num_eps += 1
		return self._num_eps
	
	def sample(self):
		"""
		Sample a training batch: transitions (obs, action, next_obs) with value / policy goals and the goal-reaching
		reward, plus the agent-specific targets (policy subgoal, SD target, LAM pair, world-model chunk).
		"""
		if self._lazy:
			# sample episodes and a random window offset into each (as the SliceSampler does)
			stride = self.cfg.data_episode_length + 1
			ep_idxs = torch.randint(self._num_eps, size=(self.cfg.batch_size,), device=self._storage_device)
			offsets = torch.randint(stride - self.cfg.data_episode_length + 1, size=(self.cfg.batch_size,), device=self._storage_device)
			episodes = _EpisodeBatch(self._data, ep_idxs * stride + offsets)
		else:
			# sample full episodes
			episodes = self._buffer.sample().view(-1, self.cfg.data_episode_length)
		# upper bound for start index: need idxs + max_offset within episode
		max_offset = self.horizon
		if self.cfg.agent == "telawm":
			max_offset = max(max_offset, self.cfg.telawm["lam_timescale"])
		if self.cfg.agent in ["acwm", "lacwm"]:
			agent_cfg = self.cfg.acwm if self.cfg.agent == "acwm" else self.cfg.lacwm
			max_offset = max(max_offset, self.horizon - 1 + agent_cfg["wm_timescale"])
		idxs = torch.randint(self.cfg.data_episode_length - max_offset, size=(self.cfg.batch_size,), device=self._storage_device)
		batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device).unsqueeze(1)  # shape: (batch_size, 1)
		slice_idxs = idxs.unsqueeze(1) + torch.arange(self.horizon, device=self._storage_device).unsqueeze(0)
		# sample policy goal indices
		policy_goal_batch_idxs, policy_goal_idxs = self._sample_goal(idxs + self.horizon - 1, self.cfg.p_curgoal_policy, self.cfg.p_trajgoal_policy, self.cfg.p_randgoal_policy, self.cfg.geom_sample_policy)
		# sample value goal indices
		value_goal_batch_idxs, value_goal_idxs = self._sample_goal(idxs + self.horizon - 1, self.cfg.p_curgoal_value, self.cfg.p_trajgoal_value, self.cfg.p_randgoal_value, self.cfg.geom_sample_value)
		# retrieve data
		obs = episodes.get('obs')[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
		action = episodes.get('action')[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
		next_obs = episodes.get('obs')[batch_idxs, slice_idxs + 1].contiguous().to(self._device, non_blocking=True)
		value_goal = episodes.get('obs')[value_goal_batch_idxs, value_goal_idxs].unsqueeze(1).repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape)).contiguous().to(self._device, non_blocking=True)
		policy_goal = episodes.get('obs')[policy_goal_batch_idxs, policy_goal_idxs].unsqueeze(1).repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape)).contiguous().to(self._device, non_blocking=True)
		reward, done = self._compute_reward(episodes, idxs, batch_idxs, slice_idxs, value_goal_batch_idxs, value_goal_idxs)

		# IQL n-step: collapse trajectory window into a single transition for the agent
		if self.cfg.agent == "iql" and self.horizon > 1:
			r = reward.reshape(self.cfg.batch_size, self.horizon, -1)
			d = done.reshape(self.cfg.batch_size, self.horizon, -1)
			discounts = (self.cfg.gamma ** torch.arange(self.horizon, device=self._device, dtype=r.dtype)).view(1, -1, 1)
			not_done = 1.0 - d
			alive = torch.ones_like(d)
			alive[:, 1:] = torch.cumprod(not_done[:, :-1], dim=1)
			reward = (r * discounts * alive).sum(dim=1, keepdim=True)  # (B, 1, 1)
			done = 1.0 - not_done.prod(dim=1, keepdim=True)  # (B, 1, 1)
			obs = obs[:, :1]
			action = action[:, :1]
			next_obs = next_obs[:, -1:]  # bootstrap state s_{t+n}
			value_goal = value_goal[:, :1]
			policy_goal = policy_goal[:, :1]

		# save in tensordict
		td = TensorDict(
            obs=self.normalizer.normalize(obs),
			action=action,
			next_obs=self.normalizer.normalize(next_obs),
            reward=reward,
            done=done,
            value_goal=self.normalizer.normalize(value_goal),
            policy_goal=self.normalizer.normalize(policy_goal),
            batch_size=(obs.shape[0], obs.shape[1]),
            device=self._device,
        )
		
		last_idx = self.cfg.data_episode_length - 1

		# Policy subgoal: the state subgoal_k steps ahead, used by saw / hiql / sd, and by the low-level policy of
		# iql / telawm / acwm / lacwm with pi_subgoal. When the policy goal lies on the same episode the subgoal is capped
		# at it (min(t+k, g)); random goals come from other episodes, so there it is capped at the episode end instead.
		uses_policy_subgoal = (
			self.cfg.agent in ["saw", "hiql", "sd"]
			or (self.cfg.agent in ["iql", "telawm", "acwm", "lacwm"] and self.cfg.pi_subgoal)
		)
		if uses_policy_subgoal:
			subgoal_idxs = slice_idxs + self.cfg.subgoal_k
			same_episode = policy_goal_batch_idxs.unsqueeze(1) == batch_idxs
			subgoal_idxs = torch.where(
				same_episode,
				torch.minimum(subgoal_idxs, policy_goal_idxs.unsqueeze(1)),
				subgoal_idxs.clamp(max=last_idx),
			)
			policy_subgoal = self._obs_at(episodes, batch_idxs, subgoal_idxs)
			if self.cfg.agent == "iql" and self.horizon > 1:
				policy_subgoal = policy_subgoal[:, :1]  # n-step IQL keeps only the first transition (see above)
			td["policy_subgoal"] = policy_subgoal

		# SD diffusion target, used when sd.goal_conditioned=false: the state subgoal_k steps ahead on the same episode,
		# independent of the sampled goals.
		if self.cfg.agent == "sd":
			sd_subgoal_idxs = (slice_idxs + self.cfg.subgoal_k).clamp(max=last_idx)
			td["sd_next_obs"] = self._obs_at(episodes, batch_idxs, sd_subgoal_idxs)

		# TeLAWM latent action model pairs (s_t, s_{t+k}), k = lam_timescale, with s_t the window's first observation.
		if self.cfg.agent == "telawm":
			lam_idxs = (idxs + self.cfg.telawm["lam_timescale"]).clamp(max=last_idx).unsqueeze(1)
			td["lam_next_obs"] = self._obs_at(episodes, batch_idxs, lam_idxs)  # (batch_size, 1, obs_dim)

		# ACWM / LACWM world-model pairs: the action chunk a_{t:t+K-1} and the state s_{t+K} it leads to (K = wm_timescale).
		# No clamping needed: max_offset leaves K steps after every slice index.
		if self.cfg.agent in ["acwm", "lacwm"]:
			agent_cfg = self.cfg.acwm if self.cfg.agent == "acwm" else self.cfg.lacwm
			wm_timescale = agent_cfg["wm_timescale"]
			td["wm_next_obs"] = self._obs_at(episodes, batch_idxs, slice_idxs + wm_timescale)  # (batch_size, horizon, obs_dim)
			chunk_idxs = slice_idxs.unsqueeze(-1) + torch.arange(wm_timescale, device=self._storage_device)  # (batch_size, horizon, K)
			td["wm_action_chunks"] = (
				episodes.get('action')[batch_idxs.unsqueeze(-1), chunk_idxs].contiguous().to(self._device, non_blocking=True)
			)  # (batch_size, horizon, K, action_dim)

		return td.squeeze(1)  # NOTE: this is to remove the extra 'horizon' dimension when self.horizon=1
	
	def _obs_at(self, episodes, batch_idxs, time_idxs):
		"""Normalized observations at [batch_idxs, time_idxs] of the sampled episodes, on the training device."""
		obs = episodes.get('obs')[batch_idxs, time_idxs].contiguous().to(self._device, non_blocking=True)
		return self.normalizer.normalize(obs)

	def _sample_goal(self, idxs, p_curgoal, p_trajgoal, p_randgoal, geom_sample):
		"""
		(episode, time) indices of a goal for each batch element: the next state (fraction p_curgoal), a future state
		of the same episode (p_trajgoal; geometric or uniform offset), or a random state (p_randgoal).
		"""
		# random goals
		randgoal_idxs = torch.randint(self.cfg.data_episode_length - 1, size=(self.cfg.batch_size,), device=self._storage_device)
		randgoal_batch_idxs = torch.randint(self.cfg.batch_size, size=(self.cfg.batch_size,), device=self._storage_device)

		# goals from the same episode
		if geom_sample:
			offsets = self._geometric_dist.sample((self.cfg.batch_size,)).to(self._storage_device, non_blocking=True)
			trajgoal_idxs = torch.minimum(idxs + offsets.squeeze(-1), torch.full_like(idxs, self.cfg.data_episode_length-1)).to(torch.int)
		else:
			distances = torch.rand(self.cfg.batch_size, device=self._storage_device)  # in [0, 1)
			trajgoal_idxs = torch.round(((idxs + 1) * distances + (self.cfg.data_episode_length - 1) * (1 - distances))).to(torch.int)
		
		# goals at the next state
		curgoal_idxs = idxs + 1

		batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device)
		goal_batch_idxs = torch.where(batch_idxs >= self.cfg.batch_size * (1-p_randgoal), randgoal_batch_idxs, batch_idxs)
		
		goal_idxs = torch.where(batch_idxs < self.cfg.batch_size * p_curgoal, curgoal_idxs, trajgoal_idxs)
		goal_idxs = torch.where(batch_idxs >= self.cfg.batch_size * (1-p_randgoal), randgoal_idxs, goal_idxs)

		return goal_batch_idxs, goal_idxs

	def _compute_reward(self, episodes, idxs, batch_idxs, slice_idxs, goal_batch_idxs, goal_idxs):
		"""
		Goal-reaching reward (0 at the goal, -1 otherwise) and done flag for each transition: from the environment's
		success check (reward=env) or from the next state being the goal state itself (reward=unsupervised).
		"""
		if self.cfg.reward == 'unsupervised':
			# self-supervised reward
			episode_id = episodes.get('episode')[batch_idxs.squeeze(1), idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
			goal_episode_id = episodes.get('episode')[goal_batch_idxs, idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
			goal_idxs = goal_idxs.unsqueeze(1).repeat(1, self.horizon).contiguous()
			reward = torch.where((episode_id == goal_episode_id) & (slice_idxs+1 == goal_idxs), 0.0, -1.0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
			done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)

		else:
			qpos = episodes.get('qpos')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
			goal_qpos = episodes.get('qpos')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1).contiguous().to(self._device, non_blocking=True)
			# reshape horizon to be part of the batch size
			qpos = qpos.view(self.cfg.batch_size * self.horizon, *qpos.shape[2:])
			goal_qpos = goal_qpos.view(self.cfg.batch_size * self.horizon, *goal_qpos.shape[2:])
			if 'scene' in self.cfg.task:
				button_states = episodes.get('button_states')[batch_idxs, slice_idxs+1].contiguous().to(self._device, non_blocking=True)
				goal_button_states = episodes.get('button_states')[goal_batch_idxs, goal_idxs].unsqueeze(1).repeat(1, self.horizon, 1).contiguous().to(self._device, non_blocking=True)
				# reshape button states to be part of the batch size
				button_states = button_states.view(self.cfg.batch_size * self.horizon, *button_states.shape[2:])
				goal_button_states = goal_button_states.view(self.cfg.batch_size * self.horizon, *goal_button_states.shape[2:])
			else:
				button_states = None
				goal_button_states = None
			# compute reward
			reward = self.env.compute_reward_batch(qpos, goal_qpos, button_states, goal_button_states)
			# reshape reward back to (bs, horizon, 1)
			reward = reward.view(self.cfg.batch_size, self.horizon, 1)
			done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)
			
		done = done if self.cfg.done_signal else torch.zeros_like(reward)

		return reward, done
	
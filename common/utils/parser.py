import dataclasses
import re
from pathlib import Path
from typing import Any
import random

import hydra
from omegaconf import OmegaConf


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Seed
	if cfg.seed == 'random':
		cfg.seed = random.randint(0, 50_000)

	# Environment
	assert cfg.reward in ['unsupervised', 'env']
	
	assert cfg.obs in ['state', 'rgb', 'latent']
	
	# Agent
	assert cfg.agent in ['iql', 'tdmpc', 'saw', 'hiql', 'telawm', 'acwm', 'lacwm', 'sd', 'ae']
	agent_cfg = {'iql': cfg.iql, 'tdmpc': cfg.tdmpc, 'saw': cfg.saw, 'hiql': cfg.hiql, 'telawm': cfg.telawm, 'acwm': cfg.acwm, 'lacwm': cfg.lacwm, 'sd': cfg.sd, 'ae': cfg.ae}.get(cfg.agent, {})
	cfg.vf_chkpt = agent_cfg.get("vf_chkpt")
	cfg.policy_chkpt = agent_cfg.get("policy_chkpt")
	cfg.tau = agent_cfg.get("tau")
	cfg.alpha = agent_cfg.get("alpha")
	cfg.expectile = agent_cfg.get("expectile")
	cfg.pi_extract_type = agent_cfg.get("pi_extract_type")
	cfg.pi_subgoal = agent_cfg.get("pi_subgoal")
	cfg.subgoal_k = agent_cfg.get("subgoal_k")
	cfg.subgoal_steps = agent_cfg.get("subgoal_steps")
	
	# Paths
	cfg.work_dir = Path(hydra.utils.get_original_cwd())
	cfg.data_dir = Path(hydra.utils.get_original_cwd()) / f'{cfg.data_dir}'
	# Optional run_suffix keeps runs of the same agent/task/seed in distinct dirs: <agent>_<task>_<seed>[_<run_suffix>]
	_run_suffix = cfg.get("run_suffix", "") or ""
	_run_name = f'{cfg.agent}_{cfg.task}_{str(cfg.seed)}' + (f'_{_run_suffix}' if _run_suffix else '')
	cfg.log_dir = Path(hydra.utils.get_original_cwd()) / 'output' / 'train' / _run_name
	cfg.checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.checkpoint}'
	cfg.vf_chkpt = Path(hydra.utils.get_original_cwd()) / f'{cfg.vf_chkpt}'
	cfg.policy_chkpt = Path(hydra.utils.get_original_cwd()) / f'{cfg.policy_chkpt}'
	cfg.eval_log_dir = Path(hydra.utils.get_original_cwd()) / 'output' / 'eval' / _run_name
	cfg.eval_checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.eval_checkpoint}'
	# frozen image autoencoder (obs=latent, and the latent high levels over a pixel low level)
	if cfg.obs == 'latent':
		assert cfg.get("latent_encoder_checkpoint"), "obs=latent requires latent_encoder_checkpoint (an autoencoder checkpoint)"
	if cfg.get("latent_encoder_checkpoint"):
		cfg.latent_encoder_checkpoint = Path(hydra.utils.get_original_cwd()) / f'{cfg.latent_encoder_checkpoint}'
	
	# Convenience
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.exp_name = f"{cfg.agent}_{cfg.task}"

	return cfg_to_dataclass(cfg)

import hashlib
import os
from tqdm import trange

import numpy as np
import torch
from tensordict.tensordict import TensorDict

import ogbench

from common.buffer import Buffer


def latent_dataset_path(cfg):
    """Cache path of the encoded dataset, keyed by a digest of the encoder checkpoint."""
    digest = hashlib.sha1(open(cfg.latent_encoder_checkpoint, 'rb').read()).hexdigest()[:8]
    return os.path.join(cfg.data_dir, f'{cfg.task}-latent-{digest}.npz')


def encode_and_save_dataset(cfg, ae, save_path):
    """Encode the image dataset with the frozen autoencoder and cache it. All other keys (incl. qpos for
    the reward) are kept unchanged."""
    np_dataset, _ = ogbench.make_env_and_datasets(
        cfg.task, dataset_dir=cfg.data_dir, dataset_only=True, compact_dataset=True, add_info=True,
    )
    obs = torch.from_numpy(np_dataset["observations"])
    assert obs.dtype == torch.uint8 and obs.ndim == 4, (
        f"obs=latent encodes image observations; got {tuple(obs.shape)} of {obs.dtype}"
    )

    latents = []
    chunk_size = 1024
    print(f"Encoding {len(obs):,} frames with the autoencoder...")
    with torch.no_grad():
        for i in trange(0, len(obs), chunk_size):
            z = ae.encode(obs[i:i + chunk_size].to('cuda:0'))
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0).astype(np.float32)

    saved = {'observations': latents}
    for key in np_dataset:
        if key != 'observations':
            saved[key] = np_dataset[key]

    # write-then-rename, so a killed job never leaves a partial cache
    tmp_path = f'{save_path}.tmp.{os.getpid()}.npz'
    np.savez(tmp_path, **saved)
    os.replace(tmp_path, save_path)


def load_latent_dataset(cfg, ae):
    """The task's dataset encoded with the frozen autoencoder, built and cached on first use."""
    fp = latent_dataset_path(cfg)
    if not os.path.exists(fp):
        print(f"Latent dataset {fp} not found, encoding {cfg.task} with {cfg.latent_encoder_checkpoint}...")
        encode_and_save_dataset(cfg, ae, fp)
        print(f"Encoded dataset saved to: {fp}")
    else:
        print(f"Loading encoded dataset from {fp}")
    return dict(np.load(fp))


def load_dataset_to_buffer(cfg, env):
    """Load dataset to buffer for offline training."""
    # Load numpy dataset
    
    if cfg.obs == 'latent':
        np_dataset = load_latent_dataset(cfg, env.ae)

    else:
        np_dataset, _ = ogbench.make_env_and_datasets(cfg.task, dataset_dir=cfg.data_dir, dataset_only=True, compact_dataset=True, add_info=(cfg.reward != 'unsupervised'))
    
    # Get dataset attributes
    capacity = np_dataset["terminals"].size
    cfg.data_episode_length = int(np.nonzero(np_dataset["terminals"])[0][0]) + 1
    
    # Convert dataset to torch tensordict
    obs = torch.from_numpy(np_dataset["observations"])
    
    dataset = TensorDict({
            "obs": obs,
            "action": torch.from_numpy(np_dataset["actions"]),
        }, batch_size=capacity)
    
    # Add information for reward calculation to tensordict
    if cfg.reward != 'unsupervised':
        dataset["qpos"] = torch.from_numpy(np_dataset["qpos"])
        if 'scene' in cfg.task:
            dataset["button_states"] = torch.from_numpy(np_dataset["button_states"])
            
    # Create buffer for sampling
    buffer = Buffer(cfg, env, capacity)
    buffer.load(dataset)
    
    expected_episodes = capacity // (cfg.data_episode_length + 1)
    if buffer.num_eps != expected_episodes:
        print(f'WARNING: buffer has {buffer.num_eps} episodes, expected {expected_episodes} episodes for {cfg.task} task.')

    return buffer

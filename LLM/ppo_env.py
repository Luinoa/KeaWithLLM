# ppo_env.py
# TODO: Rewirte this file to fit Kea environment
import gym
import torch
import numpy as np

def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return thunk

class EnvRunner:
    def __init__(self, args, run_name, env_params, device):
        self.args = args
        self.run_name = run_name
        self.env_params = env_params
        self.device = device
        self.envs = self._create_envs()

    def _create_envs(self):
        from gym.vector import SyncVectorEnv
        env_fns = [make_env(self.args.env_id, self.args.seed + i, i,
                              self.args.capture_video, self.run_name, self.env_params)
                   for i in range(self.args.num_envs)]
        return SyncVectorEnv(env_fns)

    def reset(self):
        obs = self.envs.reset()
        return torch.tensor(obs, device=self.device)

    def step(self, actions):
        next_obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        next_obs = torch.tensor(next_obs, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        return next_obs, rewards, dones, infos

    def close(self):
        self.envs.close()

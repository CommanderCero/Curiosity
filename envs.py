import numpy as np
import cv2
import gym
import gym_super_mario_bros

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
import stable_baselines3.common.atari_wrappers as atari_wrappers

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def make_env(env_id, num_envs, seed=0) -> gym.Env:
    if env_id.startswith("SuperMarioBros"):
        env_fn = lambda: make_mario_env(env_id)
        env = make_vec_env(env_fn, n_envs=num_envs, seed=seed)
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)
        return env
    elif env_id.startswith("ALE"):
        env = make_atari_env(env_id, n_envs=num_envs, seed=seed)
        env = VecTransposeImage(env)
        return env
    
    return make_vec_env(env_id, n_envs=num_envs, seed=seed)
    
def make_mario_env(env_id) -> gym.Env:
    env = gym_super_mario_bros.make(env_id)
    env = MarioWrapper(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

class OneLiveLevelOnly(gym.Wrapper):
    def __init__(self, env):
        super(OneLiveLevelOnly, self).__init__(env)
        self.prev_lives = None
        self.level = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.prev_lives is None:
            self.prev_lives = info['life']
            self.level = (info['stage'], info['world'])
        elif info['life'] < self.prev_lives:
            done = True
        elif (info['stage'], info['world']) != self.level:
            done = True
            
        return obs, reward, done, info
    
    def reset(self):
        self.prev_lives = None
        self.level = None
        return self.env.reset()

class MarioWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, screen_size: int = 84):
        env = OneLiveLevelOnly(env)
        env = atari_wrappers.WarpFrame(env, width=screen_size, height=screen_size)
        env = atari_wrappers.ClipRewardEnv(env)
        super().__init__(env)
        
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state / 255.0, reward, done, info
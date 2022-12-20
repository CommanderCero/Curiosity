import models
import gym

import torch
import torch.nn as nn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnvWrapper, VecTransposeImage
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.env_util import make_atari_env


class CuriosityRewardEnv(VecEnvWrapper):
    def __init__(self, venv, embedding_net, forward_model):
        super().__init__(venv)
        self.embedding_net = embedding_net
        self.forward_model = forward_model
        self.last_actions = None
        self.last_obs = None
    
    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs
    
    def step_async(self, actions):
        self.venv.step_async(actions)
        self.last_actions = actions

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        
        self.last_obs = obs
        return obs, reward, done, info
    
    def create_train_callback(self, inverse_forward_model):
        return TrainCuriosityCallback(self.embedding_net, self.forward_model, inverse_forward_model)
        
class TrainCuriosityCallback(BaseCallback):
    def __init__(self, embedding_net, forward_model, inverse_forward_model):
        super().__init__()
        self.embedding_net = embedding_net
        self.forward_model = forward_model
        self.inverse_forward_model = inverse_forward_model
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        for rollout_data in self.model.rollout_buffer.get(batch_size=None):
            states = rollout_data.observations
            embedded_states = self.embedding_net(states)
            
            #next_states = rollout_data.observations[1:]
        
            #state_embeddings = self.embedding_net(rollout_data.states # ToDo Actually do embedding!
            #next_state_embeddings = rollout_data.next_states # ToDo Actually do embedding!
            
            #next_state_predictions = self.forward_model(state_embeddings, rollout_data.actions)
            #action_predictions = self.inverse_forward_model(state_embeddings, next_state_embeddings)
        
if __name__ == "__main__":
    import gym
    from stable_baselines3.a2c import A2C
    
    # Initialize Environment
    #env = 
    #env = make_atari_env("PongNoFrameskip-v4", n_envs=4)
    #env = VecTransposeImage(env)
    env = DummyVecEnv([lambda: gym.make("CartPole-v1") for i in range(4)])
    
    # Initialize ICM Models
    assert isinstance(env.observation_space, gym.spaces.Box), "This ICM Implementation can only handle box observation spaces."
    #assert len(env.observation_space.shape) <= 2, "This ICM Implementation can only handle 1- and 2-dimensional observation spaces."
    assert isinstance(env.action_space, gym.spaces.Discrete), "This ICM Implementation can only handle discrete action spaces."
    num_actions = env.action_space.n
    
    if len(env.observation_space.shape) == 1:
        embedding_net = FlattenExtractor(env.observation_space)
        embedding_size = env.observation_space.shape[0]
    else:
        embedding_size = 128
        embedding_net = models.Conv2DEmbedding(env.observation_space.shape, embedding_size)
    
    forward_model = models.OneHotForwardModel(
        models.create_feedforward([embedding_size + num_actions, embedding_size * 2, embedding_size]),
        num_actions
    )
    inverse_forward_model = models.CategoricalActionPredictor(
        logits_net = models.create_feedforward([embedding_size * 2, 256, num_actions])
    )
    
    # Initalize ICM
    env = CuriosityRewardEnv(env, embedding_net, forward_model)
    train_curiosity_callback = env.create_train_callback(inverse_forward_model)
    
    # Initalize Model
    model = A2C("MlpPolicy", env, verbose=1, n_steps=64)
    
    # Train
    model.learn(
        total_timesteps=25000,
        callback=train_curiosity_callback
    )
import gym
import models
import numpy as np
import torch
import torch.nn as nn

from sb3_utils import VideoRecorderCallback
from envs import make_env
from typing import Dict

from stable_baselines3 import A2C
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnvWrapper, VecTransposeImage
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

class CuriosityRewardEnv(VecEnvWrapper):
    def __init__(self, venv, embedding_net, forward_model, curiosity_scalar=0.1, use_extrinsic_rewards=True):
        super().__init__(venv)
        self.embedding_net = embedding_net
        self.forward_model = forward_model
        self.curiosity_scalar = curiosity_scalar
        self.use_extrinsic_rewards = use_extrinsic_rewards
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
        observations, rewards, dones, infos = self.venv.step_wait()
        
        if not self.use_extrinsic_rewards:
            rewards[:] = 0
        
        if abs(self.curiosity_scalar) > 0.0001:
            # Compute curiosity reward, really inefficient I know
            with torch.no_grad():
                state_embeddings = self.embedding_net(torch.from_numpy(self.last_obs))
                predicted_next_states = self.forward_model(state_embeddings, torch.from_numpy(self.last_actions))
                next_state_embeddings = self.embedding_net(torch.from_numpy(observations))
                
                curiosity_reward = torch.norm(predicted_next_states - next_state_embeddings, dim=1, p=2)
                curiosity_reward = curiosity_reward.numpy()
                rewards += curiosity_reward * self.curiosity_scalar
        
        self.last_obs = observations
        return observations, rewards, dones, infos
    
    def create_train_callback(self, inverse_forward_model):
        return TrainCuriosityCallback(self.embedding_net, self.forward_model, inverse_forward_model)
        
class TrainCuriosityCallback(BaseCallback):
    def __init__(self, embedding_net, forward_model, inverse_forward_model):
        super().__init__()
        self.embedding_net = embedding_net
        self.forward_model = forward_model
        self.inverse_forward_model = inverse_forward_model
        
        self.fm_optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=0.0003)
        self.inverse_fm_optimizer = torch.optim.Adam(self.inverse_forward_model.parameters(), lr=0.0003)
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        assert isinstance(rollout_buffer, RolloutBuffer), "This Curiosity implementation is hardcoded to only work with OnPolicyAlgorithms that use a RolloutBuffer"
        states, actions, next_states = self.__collect_data(rollout_buffer)
        
        # Compute Losses
        embedded_states = self.embedding_net(states)
        embedded_next_states = self.embedding_net(next_states) # ToDo lots of duplicate computations
        
        fm_loss = self.forward_model.compute_loss(
            states=embedded_states, 
            actions=actions, 
            next_states=embedded_next_states
        )
        inverse_fm_loss = self.inverse_forward_model.compute_loss(
            states=embedded_states, 
            actions=actions, 
            next_states=embedded_next_states
        )
        loss = 0.5*fm_loss + 0.5*inverse_fm_loss
        
        # Update
        self.fm_optimizer.zero_grad()
        self.inverse_fm_optimizer.zero_grad()
        loss.backward()
        
        nn.utils.clip_grad_norm_(self.forward_model.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.inverse_forward_model.parameters(), 0.5)
        
        self.fm_optimizer.step()
        self.inverse_fm_optimizer.step()
        
        # Logging
        predicted_next_states = self.forward_model(embedded_states, actions)
        intrinsic_reward = torch.norm(predicted_next_states - embedded_next_states, dim=1, p=2)
        
        self.logger.record("curiosity/forward_model_loss", fm_loss.item())
        self.logger.record("curiosity/inverse_forward_model_loss", inverse_fm_loss.item())
        self.logger.record("curiosity/curiosity_mean", intrinsic_reward.mean().item())
        self.logger.record("curiosity/curiosity_variance", intrinsic_reward.var().item())
        
    def __collect_data(self, rollout_buffer: RolloutBuffer):
        states = rollout_buffer.swap_and_flatten(rollout_buffer.observations)
        actions = rollout_buffer.swap_and_flatten(rollout_buffer.actions)
        episode_starts = rollout_buffer.swap_and_flatten(rollout_buffer.episode_starts)
        episode_ends = np.roll(episode_starts, -1)
        
        # The rollout buffer collects data as (state, action) pairs
        # We are guaranteed that consecutive tuples ((s1, a1), (s2, a2), ...) represent one episode
        # All we have to do is to seperate them into (current_state, action, next_state)
        # The "episode_start" flag indicates that we are now in an different episode
        # So we can use that to extract the states that we care about
        current_states = states[~episode_ends.astype(bool).flatten()]
        current_actions = actions[~episode_ends.astype(bool).flatten()]
        next_states = states[~episode_starts.astype(bool).flatten()]
        
        return (
            torch.as_tensor(current_states),
            torch.as_tensor(current_actions),
            torch.as_tensor(next_states)
        )
    
class SaveModulesCallback(BaseCallback):
    def __init__(self, modules: Dict[str, nn.Module], save_path):
        super().__init__()
        self.modules = modules
        self.save_path = save_path
    
    def _on_step(self) -> bool:
        for name, module in self.modules.items():
            state_dict = module.state_dict()
            torch.save(state_dict, f"{self.save_path}/{name}.torch")
        
        return True
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Trains A2C with curiosity rewards.'
    )
    parser.add_argument("-env_id", default="SuperMarioBros-v0", help="ID of the environment.")
    parser.add_argument("-num_envs", default=4, type=int, help="How many environments to use for parallelization.")
    parser.add_argument("-train_steps", default=1000000, type=int, help="How many steps in the environment should the training take.")
    parser.add_argument("-curiosity_scalar", default=1, type=float, help="A scalar for scaling the curiosity reward. Set to 0 to remove curiosity.")
    parser.add_argument("--no_extrinsic_rewards", default=True, type=bool, help="A flag for training the agent without extrinsic rewards.")
    #parser.add_argument("-saved_model", help="Path to an model zip-file which should be trained further.")
    args = parser.parse_args()
    args.env_id = "ALE/Pong-v5"
    
    # Initialize Environment
    env = make_env(args.env_id, num_envs=args.num_envs)
    
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
    
    # Initialize Logging
    logger = configure_logger(verbose=1, tensorboard_log=f"./logs/test_no_extrinsic")
    eval_env = make_env(args.env_id, num_envs=1)
    video_recorder = VideoRecorderCallback(eval_env, render_freq=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=video_recorder,
        callback_on_new_best=SaveModulesCallback(
            {"embedding": embedding_net, "forward_model": forward_model, "inverse_forward_model": inverse_forward_model},
            save_path=f"{logger.dir}"
        ),
        eval_freq=50000//args.num_envs,
        best_model_save_path=f"{logger.dir}"
    )
    
    # Initalize ICM
    env = CuriosityRewardEnv(env, embedding_net, forward_model, curiosity_scalar=args.curiosity_scalar)
    train_curiosity_callback = env.create_train_callback(inverse_forward_model)
    
    # Initalize Model
    if len(env.observation_space.shape) == 1:
        policy_type = "MlpPolicy"
    else:
        policy_type = "CnnPolicy"
        
    model = A2C(policy_type, env, n_steps=128//args.num_envs, learning_rate=0.0003)
    model.set_logger(logger)
    
    # Train
    model.learn(
        total_timesteps=10000000,
        callback=CallbackList([
            train_curiosity_callback, 
            eval_callback
        ])
    )
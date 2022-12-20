import gym
import torch
import os

import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

from typing import Any, Dict

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

def make_env(env_id, num_cpus, seed=0):
    env = make_atari_env(env_id, n_envs=num_cpus, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Trains A2C from StableBaselines3 on Atari Environments.'
    )
    parser.add_argument("-env_id", default="PongNoFrameskip-v4", help="ID of the gym-atari environment.")
    parser.add_argument("-cpus", default=4, type=int, help="How many CPUs to use for parallelization.")
    parser.add_argument("-train_steps", default=25000, type=int, help="How many steps in the environment should the training take.")
    parser.add_argument("-saved_model", help="Path to an model zip-file which should be trained further.")
    parser.add_argument("-lr", default=0.0007, type=float, help="The learning rate to be used to train the actor critic.")
    args = parser.parse_args()
    
    # Initialize Logging
    logger = configure_logger(verbose=1, tensorboard_log=f"./logs/{args.env_id}_{args.lr}")
    
    # Initialize Model
    env = make_env(args.env_id, num_cpus=args.cpus, seed=0)
    if args.saved_model:
        model = A2C.load(args.saved_model, env=env)
    else:
        model = A2C("CnnPolicy", env, verbose=1, learning_rate=args.lr)
    model.set_logger(logger)

    # Initialize Logging
    eval_env = make_env(args.env_id, num_cpus=args.cpus, seed=0)
    video_recorder = VideoRecorderCallback(eval_env, render_freq=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=video_recorder,
        eval_freq=10000//args.cpus,
        best_model_save_path=f"{logger.dir}"
    )
    
    # Train
    model.learn(
        total_timesteps=args.train_steps,
        callback=eval_callback
    )
    env.close()

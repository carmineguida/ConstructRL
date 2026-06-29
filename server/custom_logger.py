from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np

class SummaryWriterCallback(BaseCallback):
    def __init__(self, vec_env, verbose = 0):
        super().__init__(verbose)
        self.log_freq = 1 # Log every time (this can be changed to 100, 1000, etc.)
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.rewards = np.zeros(self.log_freq, dtype=float)
        self.steps = np.zeros(vec_env.num_envs, dtype=int)


    def _on_training_start(self):
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        i = self.n_calls % self.log_freq
        self.rewards[i] = np.mean(self.vec_env.terminal_rewards)
        self.steps[i] = np.mean(self.vec_env.terminal_steps)

        if self.n_calls % self.log_freq == 0:            
            self.tb_formatter.writer.add_scalar("rewards", np.mean(self.rewards), self.n_calls)
            self.tb_formatter.writer.add_scalar("steps", np.mean(self.steps), self.n_calls)
            #for i in range(self.num_envs):
            #    self.tb_formatter.writer.add_scalar("rewards/env #{}".format(i+1), rewards[i], self.n_calls)

        return True
# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for running Atari 2600 agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time


import atari_py
from dopamine.atari import preprocessing
from dopamine.common import checkpointer
from dopamine.common import iteration_statistics
from dopamine.common import logger
from dopamine.agents.agent_utils import *
import gym
import numpy as np
import tensorflow as tf

import gin.tf

RPG_AGENTS = ['dqnrpg', 'rainbowrpg', 'implicit_quantilerpg', 'c51rpg']


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


def create_atari_environment(game_name, sticky_actions=True):
  """Wraps an Atari 2600 Gym environment with some basic preprocessing.

  This preprocessing matches the guidelines proposed in Machado et al. (2017),
  "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
  Problems for General Agents".

  The created environment is the Gym wrapper around the Arcade Learning
  Environment.

  The main choice available to the user is whether to use sticky actions or not.
  Sticky actions, as prescribed by Machado et al., cause actions to persist
  with some probability (0.25) when a new command is sent to the ALE. This
  can be viewed as introducing a mild form of stochasticity in the environment.
  We use them by default.

  Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

  Returns:
    An Atari 2600 environment with some standard preprocessing.
  """
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  env = gym.make(full_game_name)
  # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
  # handle this time limit internally instead, which lets us cap at 108k frames
  # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
  # restoring states.
  env = env.env
  env = preprocessing.AtariPreprocessing(env)
  return env


@gin.configurable
class Runner(object):
  """Object that handles running Atari 2600 experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, game_name='Pong')
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               random_seed,
               agent_name,
               game_name,
               num_iterations,
               create_environment_fn=create_atari_environment,
               sticky_actions=True,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        Atari 2600 Gym environment, and returns an agent.
      create_environment_fn: A function which receives a game name and creates
        an Atari 2600 Gym environment.
      game_name: str, name of the Atari 2600 domain to run.
      sticky_actions: bool, whether to enable sticky actions in the environment.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      evaluation_steps: int, the number of evaluation steps to perform.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    assert game_name is not None
    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)
    self.average_reward_eval = -100
    self.game_name = game_name
    self.agent_name = agent_name

    self._environment = create_environment_fn(game_name, sticky_actions)
    # Set up a session and initialize variables.
    tf.set_random_seed(random_seed)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    self._sess = tf.Session('',
                            config=tfconfig)
    # self._sess = tf.Session('',
    #                         config=tf.ConfigProto(allow_soft_placement=True))
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)
    tf.logging.info('Running %s with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t random_seed: %s', random_seed)
    tf.logging.info('\t num_iterations: %s', num_iterations)
    tf.logging.info('\t training_steps: %s', training_steps)
    tf.logging.info('\t sticky_actions: %s', sticky_actions)
    tf.logging.info('\t game_name: %s', game_name)
    # self._sess = tf.Session('',
    #                         config=tf.ConfigProto(allow_soft_placement=True))
    # self._agent = create_agent_fn(self._sess, self._environment,
    #                               summary_writer=self._summary_writer)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # # restore from pretained model a quick fix.
    # base_restore_dir = "/mnt/research/linkaixi/AllData/pommerman/dopaminecheckpts"
    # agent_name = "c51"
    # restore_dir = base_restore_dir + "/{}/{}/1/tf_checkpoints".format(agent_name, game_name)
    # self.restore_checkpoints(restore_dir, "tf_ckpt-199")

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    # self._checkpoint_dir = base_dir + "/checkpoints"
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        assert 'logs' in experiment_data
        assert 'current_iteration' in experiment_data
        self._logger.data = experiment_data['logs']
        self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def restore_checkpoints(self, restore_dir, filename):
    saver = tf.train.Saver()
    saver.restore(self._sess, os.path.join(restore_dir, filename))

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action: int, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action: int, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward)

  def _end_episode_store(self, reward, total_reward, step_number, is_opt):
    """Finalizes an episode run and store optimal trajectories.

    Args:
      reward: float, the last reward from the environment.
    """
    if is_opt:   # if it is optimal trajectories, then store it.
      self._agent.end_episode_(reward, total_reward, step_number)
    else:   # else only store for DQN
      self._agent.end_episode(reward)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      # reward = np.clip(reward, -1, 1)  todo

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)
    if self.agent_name in RPG_AGENTS:
      is_opt = False
      if total_reward >= episodic_return[self.game_name]:
        is_opt = True
      self._end_episode_store(reward, total_reward, step_number, is_opt)
    else:
      self._end_episode(reward)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

    Args:
      min_steps: int, minimum number of steps to generate in this phase.
      statistics: `IterationStatistics` object which records the experimental
        results.
      run_mode_str: str, describes the run mode for this agent.

    Returns:
      Tuple containing the number of steps taken in this phase (int), the sum of
        returns (float), and the number of episodes performed (int).
    """
    step_count = 0
    num_episodes = 0
    sum_returns = 0.
    num_good_trajs = 0
    good_traj_label = 0
    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()

      good_traj_label = 0
      if episode_return >= episodic_return[self.game_name]:
        good_traj_label = 1
        num_good_trajs += 1
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return,
          '{}_episode_goodtraj'.format(run_mode_str): good_traj_label
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
      # We use sys.stdout.write instead of tf.logging so as to flush frequently
      # without generating a line break.

      if self.agent_name in ['rpg', 'repg']:
        sys.stdout.write('epsilon: {} '.format(self._agent.epsilon_current) +
                         'replaysize {}\r'.format(self._agent.current_replay_size))
      elif self.agent_name in RPG_AGENTS:
        sys.stdout.write('Opt replay size: {} '.format(self._agent._replay_opt.memory.add_count))

      sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}'.format(episode_return) +
                       'Good traj?: {}\r'.format(good_traj_label))
      sys.stdout.flush()
    return step_count, sum_returns, num_episodes, num_good_trajs

  def _run_train_phase(self, statistics, eval_mode=False):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: The average reward generated in this phase.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = eval_mode
    start_time = time.time()
    number_steps, sum_returns, num_episodes, num_good_trajs = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    average_good_trajs = num_good_trajs / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_goodtraj': average_good_trajs})
    time_delta = time.time() - start_time
    tf.logging.info('Average undiscounted return per training episode: %.2f',
                    average_return)
    tf.logging.info('Average training steps per second: %.2f',
                    number_steps / time_delta)
    return num_episodes, average_return

  def _run_eval_phase(self, statistics):
    """Run evaluation phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes: int, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
    """
    # Perform the evaluation phase -- no learning.
    self._agent.eval_mode = True
    _, sum_returns, num_episodes, _ = self._run_one_phase(
        self._evaluation_steps, statistics, 'eval')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                    average_return)
    statistics.append({'eval_average_return': average_return})
    return num_episodes, average_return

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)

    train_eval_mode = False
    # if self.game_name == "Pong":
    if self.average_reward_eval >= episodic_return_switch[self.game_name]:
      train_eval_mode = True
      print("Stop training at iteration {}".format(iteration))

    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics, train_eval_mode)
    # else:
    #   # don't train, only for evaluation.
    #   num_episodes_train, average_reward_train = 0, 0

    if self.agent_name in RPG_AGENTS and self._agent._replay_opt.memory.add_count == 0:
      num_episodes_eval, average_reward_eval = -10000, -10000
      # if we didn't train rpg, don't waste time evaluate it.
    else:
      num_episodes_eval, average_reward_eval = self._run_eval_phase(
          statistics)
    self.average_reward_eval = average_reward_eval
    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_train: int, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes',
                         simple_value=num_episodes_train),
        tf.Summary.Value(tag='Train/AverageReturns',
                         simple_value=average_reward_train),
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running Atari 2600 experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        Atari 2600 Gym environment, and returns an agent.
    """
    tf.logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(
        base_dir=base_dir, create_agent_fn=create_agent_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward):
    """Save statistics as tensorboard summaries."""
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
    ])
    self._summary_writer.add_summary(summary, iteration)

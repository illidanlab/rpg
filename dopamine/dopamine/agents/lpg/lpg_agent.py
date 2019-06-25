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
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from dopamine.agents.agent_utils import *
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf
from tensorflow.distributions import Categorical

import gin.tf
from collections import deque

slim = tf.contrib.slim

NATURE_DQN_OBSERVATION_SHAPE = (84, 84)  # Size of downscaled Atari 2600 frame.
NATURE_DQN_DTYPE = tf.uint8  # DType of Atari 2600 observations.
NATURE_DQN_STACK_SIZE = 4  # Number of frames in the state stack.


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.

    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
      Use epsilon from there on.

    Args:
      decay_period: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      epsilon: float, the final value to which to decay the epsilon parameter.

    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


@gin.configurable
class LPGAgent(object):
    """An implementation of the DQN agent."""

    def __init__(self,
                 sess,
                 num_actions,
                 game_name="Pong",
                 observation_shape=NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=NATURE_DQN_DTYPE,
                 stack_size=NATURE_DQN_STACK_SIZE,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 margin=1,
                 tf_device='/cpu:*',
                 use_staging=True,
                 max_tf_checkpoints_to_keep=3,
                 optimizer=tf.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     momentum=0.0,
                     epsilon=0.00001,
                     centered=True),
                 summary_writer=None,
                 summary_writing_frequency=500):
        """Initializes the agent and constructs the components of its graph.

        Args:
          sess: `tf.Session`, for executing ops.
          num_actions: int, number of actions the agent can take at any state.
          observation_shape: tuple of ints describing the observation shape.
          observation_dtype: tf.DType, specifies the type of the observations. Note
            that if your inputs are continuous, you should set this to tf.float32.
          stack_size: int, number of frames to use in state stack.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          min_replay_history: int, number of transitions that should be experienced
            before the agent begins training its value function.
          update_period: int, period between DQN updates.
          target_update_period: int, update period for the target network.
          epsilon_fn: function expecting 4 parameters:
            (decay_period, step, warmup_steps, epsilon). This function should return
            the epsilon value used for exploration during training.
          epsilon_train: float, the value to which the agent's epsilon is eventually
            decayed during training.
          epsilon_eval: float, epsilon used when evaluating the agent.
          epsilon_decay_period: int, length of the epsilon decay schedule.
          tf_device: str, Tensorflow device on which the agent's graph is executed.
          use_staging: bool, when True use a staging area to prefetch the next
            training batch, speeding training up by about 30%.
          max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
            keep.
          optimizer: `tf.train.Optimizer`, for training the value function.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
        """
        assert isinstance(observation_shape, tuple)
        tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t update_horizon: %f', update_horizon)
        tf.logging.info('\t min_replay_history: %d', min_replay_history)
        tf.logging.info('\t update_period: %d', update_period)
        # tf.logging.info('\t random_seed: %d', random_seed)
        tf.logging.info('\t epsilon_train: %f', epsilon_train)
        tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
        tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
        tf.logging.info('\t tf_device: %s', tf_device)
        tf.logging.info('\t use_staging: %s', use_staging)
        tf.logging.info('\t optimizer: %s', optimizer)
        tf.logging.info('\t game: %s', game_name)

        self.game_name = game_name
        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.stack_size = stack_size
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.cumulative_gamma = math.pow(gamma, update_horizon)
        self.min_replay_history = min_replay_history
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_eval = epsilon_eval
        self.epsilon_decay_period = epsilon_decay_period
        self.update_period = update_period
        self.eval_mode = False
        self.training_steps = 0
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency
        self.margin = margin
        self.start_training = 1000
        # todo task specific FOR PONG IS 1000 IF THIS IS TOO SMALL WE END UP WITH A DETERMINISTIC POLICY QUCKKLY
        self.highest_reward = 6    # todo task specific
        self.isPrinted = False
        self.current_replay_size = 0
        self.epsilon_current = 1

        with tf.device(tf_device):
            # Create a placeholder for the state input to the DQN network.
            # The last axis indicates the number of consecutive frames stacked.
            state_shape = (1,) + self.observation_shape + (stack_size,)
            self.state = np.zeros(state_shape)
            self.state_ph = tf.placeholder(self.observation_dtype, state_shape,
                                           name='state_ph')
            self._replay = self._build_replay_buffer(use_staging)

            self._build_networks()

            self._train_op = self._build_train_op()

        self.replay_buffer = ReplayBufferRegular(100000)

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()
        self._sess = sess
        self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

        # Variables to be initialized by the agent once it interacts with the
        # environment.
        self._observation = None
        self._last_observation = None

    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('DQN_network', ['q_values'])

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          state: `tf.Tensor`, contains the agent's current state.

        Returns:
          net: _network_type object containing the tensors output by the network.
        """
        net = tf.cast(state, tf.float32)
        net = tf.div(net, 255.)
        net = slim.conv2d(net, 32, [8, 8], stride=4)
        net = slim.conv2d(net, 64, [4, 4], stride=2)
        net = slim.conv2d(net, 64, [3, 3], stride=1)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 512)
        q_values = slim.fully_connected(net, self.num_actions, activation_fn=None)
        return self._get_network_type()(q_values)

    def _build_networks(self):
        """Builds the Q-value network computations needed for acting and training.

        These are:
          self.online_convnet: For computing the current state's Q-values.
          self.target_convnet: For computing the next state's target Q-values.
          self._net_outputs: The actual Q-values.
          self._q_argmax: The action maximizing the current state's Q-values.
          self._replay_net_outputs: The replayed states' Q-values.
          self._replay_next_target_net_outputs: The replayed next states' target
            Q-values (see Mnih et al., 2015 for details).
        """
        # Calling online_convnet will generate a new graph as defined in
        # self._get_network_template using whatever input is passed, but will always
        # share the same weights.
        self.online_convnet = tf.make_template('Online', self._network_template)

        self._net_outputs = self.online_convnet(self.state_ph)
        # using a deep network, but may affect performance with a linear
        # approximation scheme.
        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

        self._replay_net_outputs = self.online_convnet(self._replay.states)
        # treat self._net_outputs.q_values as logits
        self.logsoftmaxprob = tf.nn.log_softmax(self._net_outputs.q_values)
        self.sample = Categorical(logits=self.logsoftmaxprob).sample(1)


    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent.

        Args:
          use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.

        Returns:
          A WrapperReplayBuffer object.
        """
        return circular_replay_buffer.WrappedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """

        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
        logits = self._replay_net_outputs.q_values
        self.logsoftmaxprob = tf.nn.log_softmax(logits)
        self.neglogprob = - tf.reduce_sum(self.logsoftmaxprob * replay_action_one_hot, axis=1)
        # self.temp_loss = self.neglogprob # * self.y_pl
        loss = self.actor_loss = tf.reduce_mean(self.neglogprob)
        self.replay_action_one_hot = replay_action_one_hot

        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('hingeLoss', loss)
        return self.optimizer.minimize(loss)

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          int, the selected action.
        """
        self._reset_state()
        self._record_observation(observation)

        if not self.eval_mode:
            self._train_step()

        self.action = self._select_action()
        return self.action

    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)

        if not self.eval_mode:
            # if reward < 0:
            #     self.replay_buffer.clear()
            # elif reward > 0:
            #     self.replay_buffer.add(self._last_observation, self.action, reward, False)
            #     while self.replay_buffer.size() > 0:
            #         experience = self.replay_buffer.get_sample()
            #         state, action, reward, _ = experience
            #         self._store_transition(state, action, reward, False)
            # else:
            #     self.replay_buffer.add(self._last_observation, self.action, reward, False)
            if self.game_name in ["Pong"]:
                collect_trajectory(self, reward)
            else:
                raise ValueError("collection wrong trajectory")

            self._train_step()

        self.action = self._select_action()
        if isinstance(self.action, np.ndarray):
            pass
        return self.action

    def end_episode(self, reward):
        """Signals the end of the episode to the agent.

        We store the observation of the current time step, which is the last
        observation of the episode.

        Args:
          reward: float, the last reward from the environment.
        """
        if not self.eval_mode:
            if self.game_name in ["Pong"]:
                collect_trajectory(self, reward)
            else:
                raise ValueError("collection wrong trajectory")

    def _select_action_training(self):
        """Use EPG to select action during training, """
        return self._sess.run(self.sample, {self.state_ph: self.state})[0][0]

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
           int, the selected action.
        """
        exploration = "Randomexplore"
        if exploration == "EPG":
            self.epsilon_current = 0
            self.current_replay_size = self._replay.memory.add_count
            if self.eval_mode:
                return self._sess.run(self._q_argmax, {self.state_ph: self.state})
            return self._select_action_training()

        elif exploration == "Randomexplore":
            # epsilon greedy explore.
            # epsilon = self.epsilon_eval if self.eval_mode else self.epsilon_fn(
            #     self.epsilon_decay_period,
            #     self.training_steps,
            #     self.min_replay_history,
            #     self.epsilon_train)
            if self.training_steps < self.min_replay_history:
                epsilon = 1
            else:
                epsilon = self.epsilon_train
            if self.eval_mode:
                epsilon = self.epsilon_eval
            self.epsilon_current = epsilon
            self.current_replay_size = self._replay.memory.add_count
            #
            if random.random() <= epsilon:
                # Choose a random action with probability epsilon.
                return random.randint(0, self.num_actions - 1)
            else:
                # Choose the action with highest Q-value at the current state.
                return self._sess.run(self._q_argmax, {self.state_ph: self.state})


    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.

        if self._replay.memory.add_count > self.start_training:
            if self.training_steps % self.update_period == 0:

                # debug checked.
                # _, neglogprob, logsoftmaxprob, \
                # actor_loss, replay_action_one_hot = self._sess.run([self._train_op,
                #                                                    self.neglogprob,
                #                                                    self.logsoftmaxprob,
                #                                                    self.actor_loss,
                #                                                    self.replay_action_one_hot])
                self._sess.run(self._train_op)
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps)

            # if self.training_steps % self.target_update_period == 0:
            #     self._sess.run(self._sync_qt_ops)

        self.training_steps += 1

        if (self._replay.memory.add_count > self.start_training) and self.isPrinted is False:
            print("start training at {}".format(self.training_steps))
            self.isPrinted = True

    def _record_observation(self, observation):
        """Records an observation and update state.

        Extracts a frame from the observation vector and overwrites the oldest
        frame in the state buffer.

        Args:
          observation: numpy array, an observation from the environment.
        """
        # Set current observation. We do the reshaping to handle environments
        # without frame stacking.
        observation = np.reshape(observation, self.observation_shape)
        self._observation = observation[..., 0]
        self._observation = np.reshape(observation, self.observation_shape)
        # Swap out the oldest frame with the current frame.
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ..., -1] = self._observation

    def _store_transition(self, last_observation, action, reward, is_terminal):
        """Stores an experienced transition.

        Executes a tf session and executes replay buffer ops in order to store the
        following tuple in the replay buffer:
          (last_observation, action, reward, is_terminal).

        Pedantically speaking, this does not actually store an entire transition
        since the next state is recorded on the following time step.

        Args:
          last_observation: numpy array, last observation.
          action: int, the action taken.
          reward: float, the reward.
          is_terminal: bool, indicating if the current state is a terminal state.
        """
        self._replay.add(last_observation, action, reward, is_terminal)

    def _reset_state(self):
        """Resets the agent state by filling it with zeros."""
        self.state.fill(0)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        This is used for checkpointing. It will return a dictionary containing all
        non-TensorFlow objects (to be saved into a file by the caller), and it saves
        all TensorFlow objects into a checkpoint file.

        Args:
          checkpoint_dir: str, directory where TensorFlow objects will be saved.
          iteration_number: int, iteration number to use for naming the checkpoint
            file.

        Returns:
          A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        # Call the Tensorflow saver to checkpoint the graph.
        self._saver.save(
            self._sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration_number)
        # Checkpoint the out-of-graph replay buffer.
        self._replay.save(checkpoint_dir, iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['eval_mode'] = self.eval_mode
        bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
          agent's state.

        Args:
          checkpoint_dir: str, path to the checkpoint saved by tf.Save.
          iteration_number: int, checkpoint version, used when restoring replay
            buffer.
          bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
          bool, True if unbundling was successful.
        """
        try:
            # self._replay.load() will throw a NotFoundError if it does not find all
            # the necessary files, in which case we abort the process & return False.
            self._replay.load(checkpoint_dir, iteration_number)
        except tf.errors.NotFoundError:
            return False
        for key in self.__dict__:
            if key in bundle_dictionary:
                self.__dict__[key] = bundle_dictionary[key]
        # Restore the agent's TensorFlow graph.
        self._saver.restore(self._sess,
                            os.path.join(checkpoint_dir,
                                         'tf_ckpt-{}'.format(iteration_number)))
        return True


class ReplayBufferRegular(object):
    """ for uniformly sampling.

    """

    def __init__(self, buffer_size, random_seed=1234):
        self.buffer_size = buffer_size
        self.count = 0
        # Right side of deque contains newest experience
        self.buffer = deque()
        random.seed(random_seed)
        self.ptr, self.path_start_idx = 0, 0

    def add(self, state, action, reward, terminal):
        experience = [state, action, reward, terminal]
        assert self.count < self.buffer_size
        self.buffer.append(experience)
        self.count += 1
        self.ptr += 1
        # else:
        #     self.path_start_idx -= 1
        #     self.ptr = self.buffer_size - 1
        #     self.buffer.popleft()
        #     self.buffer.append(experience)

    def get_sample(self):
        self.count -= 1
        return self.buffer.popleft()

    def size(self):
        return self.count

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.ptr = 0
        self.path_start_idx = 0

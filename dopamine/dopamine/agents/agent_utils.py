import random
from collections import deque

def collect_trajectory(agent, reward):
    """for pong """
    if reward < 0:
        agent.replay_buffer.clear()
    elif reward > 0:
        agent.replay_buffer.add(agent._last_observation, agent.action, reward, False)
        while agent.replay_buffer.size() > 0:
            experience = agent.replay_buffer.get_sample()
            state, action, reward, _ = experience
            agent._store_transition(state, action, reward, False)
    else:
        agent.replay_buffer.add(agent._last_observation, agent.action, reward, False)



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



""" Threshold of episodic return for each game """
# we only collect trajectory that has return larger than following
episodic_return = {"Pong":      21,
                   "Breakout":  200,
                   "Bowling":   80,
                   "Boxing":    100,
                   "Freeway":   33,
                   "BankHeist": 1100,
                   "Robotank":  65,
                   "Pitfall":   0,
                   "DoubleDunk":18}

# When we have the return on evaluation phase that is greater than following,
# we stop training
episodic_return_switch = {"Pong":      21,
                          "Breakout":  200,
                          "Bowling":   80,  # maximum can be more than as 93
                          "Boxing":    100,
                          "Freeway":   32,
                          "BankHeist": 1100,
                          "Robotank":  65,
                          "Pitfall": -0.1,
                          "DoubleDunk":18}

import numpy as np
import torch
import torch.multiprocessing as mp

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor

# Code based on https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/jingweiz/pytorch-distributed/blob/master/core/memories/shared_memory.py


class Memory():

    def __init__(self, memory_size, state_dim, action_dim):

        # params
        self.memory_size = memory_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        if USE_CUDA:
            self.states = torch.zeros(self.memory_size, self.state_dim).cuda()
            self.actions = torch.zeros(
                self.memory_size, self.action_dim).cuda()
            self.n_states = torch.zeros(
                self.memory_size, self.state_dim).cuda()
            self.rewards = torch.zeros(self.memory_size, 1).cuda()
            self.dones = torch.zeros(self.memory_size, 1).cuda()

        else:
            self.states = torch.zeros(self.memory_size, self.state_dim)
            self.actions = torch.zeros(self.memory_size, self.action_dim)
            self.n_states = torch.zeros(self.memory_size, self.state_dim)
            self.rewards = torch.zeros(self.memory_size, 1)
            self.dones = torch.zeros(self.memory_size, 1)

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def get_pos(self):
        return self.pos

    # Expects tuples of (state, next_state, action, reward, done)

    def add(self, datum):

        state, n_state, action, reward, done = datum

        self.states[self.pos] = FloatTensor(state)
        self.n_states[self.pos] = FloatTensor(n_state)
        self.actions[self.pos] = FloatTensor(action)
        self.rewards[self.pos] = FloatTensor([reward])
        self.dones[self.pos] = FloatTensor([done])

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):

        upper_bound = self.memory_size if self.full else self.pos
        batch_inds = torch.LongTensor(
            np.random.randint(0, upper_bound, size=batch_size))

        return (self.states[batch_inds],
                self.n_states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds])

    def get_reward(self, start_pos, end_pos):

        tmp = 0
        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):
                tmp += self.rewards[i]
        else:
            for i in range(start_pos, self.memory_size):
                tmp += self.rewards[i]

            for i in range(end_pos):
                tmp += self.rewards[i]

        return tmp

    def repeat(self, start_pos, end_pos):

        if start_pos <= end_pos:
            for i in range(start_pos, end_pos):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

        else:
            for i in range(start_pos, self.memory_size):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

            for i in range(end_pos):

                self.states[self.pos] = self.states[i].clone()
                self.n_states[self.pos] = self.n_states[i].clone()
                self.actions[self.pos] = self.actions[i].clone()
                self.rewards[self.pos] = self.rewards[i].clone()
                self.dones[self.pos] = self.dones[i].clone()

                self.pos += 1
                if self.pos == self.memory_size:
                    self.full = True
                    self.pos = 0

class Archive(list):

    def __init__(self):

        # counter
        self.cpt = 0

    def add_sample(self, sample):

        # adding the sample to the archive
        self.append(sample)
        self.cpt += 1

    def add_samples(self, samples):

        # adding the samples to the archive
        for sample in samples:
            self.add_sample(sample)

    def add_gen(self, idx, gen):

        self[idx].gens.append(gen)

    def get_size(self):
        return min(self.max_size, self.cpt)

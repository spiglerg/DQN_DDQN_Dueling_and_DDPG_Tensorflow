"""
Giacomo Spigler
"""
import numpy as np
import random
from collections import deque


class ReplayMemoryFast:
	""" Simple queue for storing and sampling of minibatches.
		This implementation has been optimized for speed by pre-allocating the buffer memory and 
		by allowing the same element to be non-unique in the sampled minibatch. In practice, however, this 
		will never happen (minibatch size ~32-128 out of 10-100-1000k memory size).
	"""
	def __init__(self, memory_size, minibatch_size):
		self.memory_size = memory_size # max number of samples to store
		self.minibatch_size = minibatch_size

		self.experience = [None]*self.memory_size  #deque(maxlen = self.memory_size)
		self.current_index = 0
		self.size = 0

	def store(self, observation, action, reward, newobservation, is_terminal):
		self.experience[self.current_index] = (observation, action, reward, newobservation, is_terminal)
		self.current_index += 1
		self.size = min(self.size+1, self.memory_size)
		if self.current_index >= self.memory_size:
			self.current_index -= self.memory_size

	def sample(self):
		""" Samples a minibatch of minibatch_size size. """
		if self.size <  self.minibatch_size:
			return []

		samples_index  = np.floor(np.random.random((self.minibatch_size,))*self.size)

		samples		= [self.experience[int(i)] for i in samples_index]

		return samples




import numpy as np
import random
from collections import deque



class ReplayMemoryFast:
	"""Simple queue with sampling for minibatches"""
	"""This implementation is faster than the deque one because we allow for the same sample to be
       returned multiple times, though in practice it will never happen (size~32 out of 10-100-1000k memory size)"""
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

	def sample(self): #samples a minibatch of given size
		if self.size <  self.minibatch_size:
			return []

		samples_index  = np.floor(np.random.random((self.minibatch_size,))*self.size)

		samples		= [self.experience[int(i)] for i in samples_index]

		return samples






class ReplayMemory:
	"""Simple queue with sampling for minibatches"""
	def __init__(self, memory_size, minibatch_size):
		self.memory_size = memory_size # max number of samples to store
		self.minibatch_size = minibatch_size

		self.experience = deque(maxlen = self.memory_size)

	def store(self, observation, action, reward, newobservation, is_terminal):
		self.experience.append( (observation, action, reward, newobservation, is_terminal) )
		if len(self.experience) > self.memory_size:
			self.experience.popleft()

	def sample(self): #samples a minibatch of given size
		if len(self.experience) <  self.minibatch_size:
			return []

		#samples_index  = random.sample(xrange(len(self.experience)), self.minibatch_size)
		#samples		= [self.experience[i] for i in samples_index]

		samples = random.sample(self.experience, self.minibatch_size)
		return samples







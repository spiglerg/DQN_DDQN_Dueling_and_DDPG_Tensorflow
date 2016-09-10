import numpy as np
import random
from collections import deque



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




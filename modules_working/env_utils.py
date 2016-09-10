import numpy as np
from scipy.misc import imresize
import random

import cv2

import gym
from gym.spaces import Box


class AtariEnvWrapper:
	"""
	Wraps OpenAI Gym Atari environments to return the last 4 processed inputs.
	Inputs are repeated if less than 4 have been recorded in the current episode.
	Input frames are preprocessed to crop the 160x160 area of interest, and are then downsamples to 84x84.
	Frames are converted to grayscale.
	The result is a 4x84x84 state
	"""

	def __init__(self, name):
		self.name = name
		self.env = gym.make(name)

		self.action_space = self.env.action_space
		self.observation_space = Box(low=0, high=255, shape=(84, 84, 4))  #self.env.observation_space

		self.frame_num = 0

		self.monitor = self.env.monitor

		self.frames = np.zeros((84, 84, 4), dtype=np.uint8)



	def seed(self, seed=None):
		return self.env._seed(seed)

	def step(self, a):
		ob, reward, done, xx = self.env.step(a)
		return self.process_frame(ob), reward, done, xx

	def reset(self):
		self.frame_num = 0
		return self.process_frame(self.env.reset())

	def render(self):
		return self.env.render()


	def process_frame(self, frame):
		state_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		state_resized = cv2.resize(state_gray,(84,110))
		gray_final = state_resized[16:100, :]

		if self.frame_num == 0:
			self.frames[:, :, 0] = gray_final
			self.frames[:, :, 1] = gray_final
			self.frames[:, :, 2] = gray_final
			self.frames[:, :, 3] = gray_final

		else:
			self.frames[:, :, 3] = self.frames[:, :, 2]
			self.frames[:, :, 2] = self.frames[:, :, 1]
			self.frames[:, :, 1] = self.frames[:, :, 0]
			self.frames[:, :, 0] = gray_final


		import matplotlib.pyplot as plt
		#plt.imsave("out0.png", self.frames[:,:,0], cmap=plt.cm.gray)
		#plt.imsave("out1.png", self.frames[:,:,1], cmap=plt.cm.gray)
		#plt.imsave("out2.png", self.frames[:,:,2], cmap=plt.cm.gray)
		#plt.imsave("out3.png", self.frames[:,:,3], cmap=plt.cm.gray)

		#plt.imsave(str(self.frame_num)+".png", gray_final, cmap=plt.cm.gray)


		self.frame_num += 1

		return self.frames.copy()



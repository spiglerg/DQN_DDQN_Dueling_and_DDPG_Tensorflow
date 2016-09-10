"""
Giacomo Spigler
"""
import numpy as np
from scipy.misc import imresize
import random

import cv2

import gym
from gym.spaces import Box


class AtariEnvWrapper:
	"""
	Wrapper for OpenAI Gym Atari environments that returns the last 4 processed frames.
	Input frames are converted to grayscale and downsampled from 120x160 to 84x112 and cropped to 84x84 around the game's main area.
	The final size of the states is 84x84x4.

	If debug=True, an OpenCV window is opened and the last processed frame is displayed. This is particularly useful when adapting the wrapper to novel environments.
	"""
	def __init__(self, env_name, debug=False):
		self.env_name = env_name
		self.debug = debug
		self.env = gym.make(env_name)

		self.action_space = self.env.action_space
		self.observation_space = Box(low=0, high=255, shape=(84, 84, 4))  #self.env.observation_space

		self.frame_num = 0

		self.monitor = self.env.monitor

		self.frames = np.zeros((84, 84, 4), dtype=np.uint8)

		if self.debug:
			cv2.startWindowThread()
			cv2.namedWindow("Game")


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
		gray_final = state_resized[16:100,:]

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

		self.frame_num += 1

		if self.debug:
			cv2.imshow('Game', gray_final)

		return self.frames.copy()



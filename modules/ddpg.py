import numpy as np
import random
import tensorflow as tf

from replay_memory import *
from networks import *
from ou_noise import *



class DDPG(object):
	"""
	Implementation of a DDPG (Deterministic Policy Gradient) agent, from Lillicrap et. al, 2015 [``Continuous control with deep reinforcement learning''].

	reward_clipping: any negative number disables rewards clipping. Positive numbers mean that the rewards will be clipped to be in [-reward_clipping, reward_clipping]
	"""
	def __init__(self, state_size,
					   action_size,
					   session,
					   summary_writer = None,
					   exploration_period = 1000,
					   minibatch_size = 32,
					   discount_factor = 0.99,
					   experience_replay_buffer = 10000,
					   target_qnet_update_frequency = 10000,
					   initial_exploration_epsilon = 1.0,
					   final_exploration_epsilon = 0.05,
					   learning_rate_actor = 0.0001,
					   learning_rate_critic = 0.001,
					   reward_clipping = -1):

		# Setup the parameters, data structures and networks
		self.state_size = state_size
		self.action_size = action_size

		self.session = session
		self.exploration_period = float(exploration_period)
		self.minibatch_size = minibatch_size
		self.discount_factor = tf.constant(discount_factor)
		self.experience_replay_buffer = experience_replay_buffer
		self.summary_writer = summary_writer
		self.reward_clipping = reward_clipping

		self.target_qnet_update_frequency = target_qnet_update_frequency

		self.initial_exploration_epsilon = initial_exploration_epsilon
		self.final_exploration_epsilon = final_exploration_epsilon


		self.experience_replay = ReplayMemoryFast(self.experience_replay_buffer, self.minibatch_size)

		self.num_training_steps = 0



		self.actor = ActorCritic_MLP(self.state_size[0], self.action_size, name="actor", actor_or_critic=0)
		self.target_actor = ActorCritic_MLP(self.state_size[0], self.action_size, name="target_actor", actor_or_critic=0)
		self.actor_optimizer =tf.train.RMSPropOptimizer(learning_rate=learning_rate_actor, decay=0.9, epsilon=0.01) 

		self.critic = ActorCritic_MLP(self.state_size[0]+self.action_size, 1, name="critic", actor_or_critic=1)
		self.target_critic = ActorCritic_MLP(self.state_size[0]+self.action_size, 1, name="target_critic", actor_or_critic=1)
		self.critic_optimizer =tf.train.RMSPropOptimizer(learning_rate=learning_rate_critic, decay=0.9, epsilon=0.01) 

		self.noise = OUNoise(self.action_size, mu=0, theta=0.15, sigma=self.initial_exploration_epsilon)



		# Setup the computational graph
		self.create_graph()


	@staticmethod
	def copy_to_target_network(source_network, target_network):
		target_network_update = []
		for v_source, v_target in zip(source_network.variables(), target_network.variables()):
			# this is equivalent to target = source
			update_op = v_target.assign(v_source)
			target_network_update.append(update_op)
		return tf.group(*target_network_update)


	def create_graph(self):
		# Pick action given state ->   action = actor( state )
		with tf.name_scope("pick_action"):
			self.state = tf.placeholder(tf.float32, (None,)+self.state_size , name="state")
			self.actions = self.actor(self.state)


		# Predict target future reward: r  +  gamma * target_critic([next_state, target_actor(next_state)])
		with tf.name_scope("estimating_future_rewards"):
			self.next_state = tf.placeholder(tf.float32, (None,)+self.state_size , name="next_state")
			self.next_state_mask = tf.placeholder(tf.float32, (None,) , name="next_state_mask") # 0 for terminal states
			self.next_actions = self.target_actor(self.next_state)
			self.next_critic_values = tf.squeeze(tf.stop_gradient(self.target_critic([self.next_state, self.next_actions]), name="next_critic_values"), [1])
			self.rewards = tf.placeholder(tf.float32, (None,) , name="rewards")

			## double dqn:  instead of reduce_max, compute next_q_values using the non-target net, and use THOSE to select the action (z1 = tf.equal(t, tf.reduce_max(t, reduction_indices=[1], keep_dims=True)) ?);  then multiply these 1-hot action vectors by the next_q_values like in the following section, to select the actual target q values!

			self.target_q_values = self.rewards + self.discount_factor * self.next_critic_values * self.next_state_mask


		# Gradient descent
		with tf.name_scope("training_critic"):
			self.action_mask = tf.placeholder(tf.float32, (None, self.action_size) , name="actions_selected") #action that was selected
			self.y = tf.squeeze( self.critic([self.state, self.action_mask]) , [1])

			## ERROR CLIPPING
			"""
			self.error = tf.abs(self.y - self.target_q_values)
			quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
			linear_part = self.error - quadratic_part
			self.loss = tf.reduce_mean( 0.5*tf.square(quadratic_part) + linear_part )
			"""
			self.loss = tf.reduce_mean(tf.square(self.y - self.target_q_values))

			critic_gradients = self.critic_optimizer.compute_gradients(self.loss, self.critic.variables())
			for i, (grad, var) in enumerate(critic_gradients):
				if grad is not None:
					critic_gradients[i] = (tf.clip_by_norm(grad, 5), var)
			self.critic_optimize = self.critic_optimizer.apply_gradients(critic_gradients)


		with tf.name_scope("training_actor"):
			self.critic_values = tf.squeeze( self.critic([self.state, self.actions]), [1] )
			self.actor_optim = -tf.reduce_mean(self.critic_values)

			tf.histogram_summary("Q values", tf.reduce_mean(self.critic_values) ) # save q-values to track learning

			actor_gradients = self.critic_optimizer.compute_gradients(self.actor_optim, self.actor.variables())
			for i, (grad, var) in enumerate(actor_gradients):
				if grad is not None:
					actor_gradients[i] = (tf.clip_by_norm(grad, 5), var)
			self.actor_optimize = self.actor_optimizer.apply_gradients(actor_gradients)


		with tf.name_scope("target_networks_update"):
			self.hard_copy_to_target_actor = DDPG.copy_to_target_network(self.actor, self.target_actor)
			self.hard_copy_to_target_critic = DDPG.copy_to_target_network(self.critic, self.target_critic)


		self.summarize = tf.merge_all_summaries()



	def store(self, state, action, reward, next_state, is_terminal):
		# rewards clipping
		if self.reward_clipping > 0.0:
			reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)

		self.experience_replay.store(state, action, reward, next_state, is_terminal)



	def action(self, state, training = False):
		"""
		If `training', compute the epsilon-greedy parameter epsilon according to the defined exploration_period, initial_epsilon and final_epsilon.
		If not `training', use a fixed testing epsilon=0.05
		"""
		if self.num_training_steps > self.exploration_period:
			epsilon = self.final_exploration_epsilon
		else:
			epsilon =  self.initial_exploration_epsilon - float(self.num_training_steps) * (self.initial_exploration_epsilon - self.final_exploration_epsilon) / self.exploration_period

		if not training:
			epsilon = 0.0 #self.final_exploration_epsilon

		self.noise.sigma = epsilon

		action = self.session.run(self.actions, {self.state:[state] } )[0]
		action = action + self.noise.noise()

		#action = np.clip(action, -2, 2)

		return action



	def train(self):
		# Copy the QNetwork weights to the Target QNetwork.
		if self.num_training_steps == 0:
			print "Training starts..."
			self.actor.copy_to(self.target_actor)
			self.critic.copy_to(self.target_critic)


		# Sample experience from replay memory
		minibatch = self.experience_replay.sample()
		if len(minibatch)==0:
			return


		# Build the bach states
		batch_states = np.asarray( [d[0] for d in minibatch] )

		batch_actions = np.asarray( [d[1] for d in minibatch] )

		batch_rewards = np.asarray( [d[2] for d in minibatch] )
		batch_newstates = np.asarray( [d[3] for d in minibatch] )

		batch_newstates_mask = np.asarray( [not d[4] for d in minibatch] )
		#"""

		"""
		##oldcode
		# bach states
		batch_states		 = np.empty((self.minibatch_size,)+self.state_size)
		batch_newstates	  = np.empty((self.minibatch_size,)+self.state_size)
		batch_actions	= np.zeros((self.minibatch_size, self.action_size))
		batch_newstates_mask = np.empty((self.minibatch_size,))
		batch_rewards		= np.empty((self.minibatch_size,))

		for i, (state, action, reward, newstate, is_terminal) in enumerate(minibatch):
			batch_states[i] = state
			batch_actions[i] = 0
			batch_actions[i][action] = 1
			batch_rewards[i] = reward
			if not is_terminal:
				batch_newstates[i] = newstate
				batch_newstates_mask[i] = 1
			else:
				batch_newstates[i] = 0
				batch_newstates_mask[i] = 0

		#"""


		# Perform training
		scores, _, _ = self.session.run([self.critic_values, self.actor_optimize, self.critic_optimize],
									  { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions} )


		if self.num_training_steps % self.target_qnet_update_frequency == 0:
			# Hard update (copy) of the weights every # iterations
			self.session.run( self.hard_copy_to_target_actor )
			self.session.run( self.hard_copy_to_target_critic )


			# Write logs
			#print 'mean critic Q in minibatch: ',np.mean(scores)

			str_ = self.session.run(self.summarize, { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions})
			self.summary_writer.add_summary(str_, self.num_training_steps)


		self.num_training_steps += 1





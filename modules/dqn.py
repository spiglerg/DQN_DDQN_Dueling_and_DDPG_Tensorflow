"""
Giacomo Spigler
"""
import numpy as np
import random
import tensorflow as tf

from replay_memory import *
from networks import *



class DQN(object):
	"""
	Implementation of a DQN agent.

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


		self.qnet = QNetworkNature(self.state_size, self.action_size, "qnet")
		self.target_qnet = QNetworkNature(self.state_size, self.action_size, "target_qnet")

		self.qnet_optimizer =tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.01) 


		self.experience_replay = ReplayMemoryFast(self.experience_replay_buffer, self.minibatch_size)

		self.num_training_steps = 0


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
		# Pick action given state ->   action = argmax( qnet(state) )
		with tf.name_scope("pick_action"):
			self.state = tf.placeholder(tf.float32, (None,)+self.state_size , name="state")

			self.q_values = tf.identity(self.qnet(self.state) , name="q_values")
			self.predicted_actions = tf.argmax(self.q_values, dimension=1 , name="predicted_actions")

			tf.histogram_summary("Q values", tf.reduce_mean(tf.reduce_max(self.q_values, 1))) # save max q-values to track learning


		# Predict target future reward: r  +  gamma * max_a'[ Q'(s') ]
		with tf.name_scope("estimating_future_rewards"):
			self.next_state = tf.placeholder(tf.float32, (None,)+self.state_size , name="next_state")
			self.next_state_mask = tf.placeholder(tf.float32, (None,) , name="next_state_mask") # 0 for terminal states
			self.next_q_values = tf.stop_gradient(self.target_qnet(self.next_state) , name="next_q_values")
			self.rewards = tf.placeholder(tf.float32, (None,) , name="rewards")
			self.next_max_q_values = tf.reduce_max(self.next_q_values, reduction_indices=[1,]) * self.next_state_mask

			## double dqn:  instead of reduce_max, compute next_q_values using the non-target net, and use THOSE to select the action (z1 = tf.equal(t, tf.reduce_max(t, reduction_indices=[1], keep_dims=True)) ?);  then multiply these 1-hot action vectors by the next_q_values like in the following section, to select the actual target q values!

			self.target_q_values = self.rewards + self.discount_factor*self.next_max_q_values


		# Gradient descent
		with tf.name_scope("optimization_step"):
			self.action_mask = tf.placeholder(tf.float32, (None, self.action_size) , name="action_mask") #action that was selected
			self.y = tf.reduce_sum( self.q_values * self.action_mask , reduction_indices=[1,])

			## ERROR CLIPPING AS IN NATURE'S PAPER
			self.error = tf.abs(self.y - self.target_q_values)
			quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
			linear_part = self.error - quadratic_part
			self.loss = tf.reduce_mean( 0.5*tf.square(quadratic_part) + linear_part )

			qnet_gradients = self.qnet_optimizer.compute_gradients(self.loss, self.qnet.variables())
			for i, (grad, var) in enumerate(qnet_gradients):
				if grad is not None:
					qnet_gradients[i] = (tf.clip_by_norm(grad, 5), var)
			self.qnet_optimize = self.qnet_optimizer.apply_gradients(qnet_gradients)


		with tf.name_scope("target_network_update"):
			self.hard_copy_to_target = DQN.copy_to_target_network(self.qnet, self.target_qnet)

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
			epsilon = 0.05

		# Execute a random action with probability epsilon, or follow the QNet policy with probability 1-epsilon.
		if random.random() <= epsilon:
			action = random.randint(0, self.action_size-1)
		else:
			action = self.session.run(self.predicted_actions, {self.state:[state] } )[0]

		return action



	def train(self):
		# Copy the QNetwork weights to the Target QNetwork.
		if self.num_training_steps == 0:
			print "Training starts..."
			self.qnet.copy_to(self.target_qnet)


		# Sample experience from replay memory
		minibatch = self.experience_replay.sample()
		if len(minibatch)==0:
			return


		# Build the bach states
		batch_states = np.asarray( [d[0] for d in minibatch] )

		actions = [d[1] for d in minibatch]
		batch_actions = np.zeros( (self.minibatch_size, self.action_size) )
		for i in xrange(self.minibatch_size):
			batch_actions[i, actions[i]] = 1

		batch_rewards = np.asarray( [d[2] for d in minibatch] )
		batch_newstates = np.asarray( [d[3] for d in minibatch] )

		batch_newstates_mask = np.asarray( [not d[4] for d in minibatch] )


		# Perform training
		scores, _, = self.session.run([self.q_values, self.qnet_optimize],
									  { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions} )


		if self.num_training_steps % self.target_qnet_update_frequency == 0:
			# Hard update (copy) of the weights every # iterations
			self.session.run( self.hard_copy_to_target )


			# Write logs
			print 'mean maxQ in minibatch: ',np.mean(np.max(scores,1))

			str_ = self.session.run(self.summarize, { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions})
			self.summary_writer.add_summary(str_, self.num_training_steps)


		self.num_training_steps += 1





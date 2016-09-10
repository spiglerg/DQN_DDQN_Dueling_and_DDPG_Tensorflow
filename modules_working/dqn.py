import numpy as np
import random
import tensorflow as tf

from collections import deque

import sys
#sys.path.append('../../')


from replay_memory import *



class QNetwork(object):
	def weight_variable(self, shape, fanin=0):
		if fanin==0:
			initial = tf.truncated_normal(shape, stddev=0.01)
		else:
			mod_init = 1.0 / math.sqrt(fanin)
			initial = tf.random_uniform( shape , minval=-mod_init, maxval=mod_init)

		return tf.Variable(initial)

	def bias_variable(self, shape, fanin=0):
		if fanin==0:
			initial = tf.constant(0.01, shape=shape)
		else:
			mod_init = 1.0 / math.sqrt(fanin)
			initial = tf.random_uniform( shape , minval=-mod_init, maxval=mod_init)

		return tf.Variable(initial)


	def variables(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)


	def copy_to(self, dst_net):
		"""
		mn = ModelNetwork(2, 3, 0, "actor")
		mn_target = ModelNetwork(2, 3, 0, "target_actor")

		s=tf.InteractiveSession()
		s.run( tf.initialize_all_variables() )

		mn.copy_to(mn_target)
		"""
		v1 = self.variables()
		v2 = dst_net.variables()

		for i in range(len(v1)):
			v2[i].assign( v1[i] ).eval()


	def print_num_of_parameters(self):
		list_vars = self.variables()
		total_parameters = 0
		for variable in list_vars:
			# shape is an array of tf.Dimension
			shape = variable.get_shape()
			variable_parametes = 1
			for dim in shape:
				variable_parametes *= dim.value
			total_parameters += variable_parametes
		print '# of parameters in network ',self.name,': ',total_parameters,'  ->  ',np.round(float(total_parameters)/1000000.0, 2),'M'







class QNetworkNIPS(QNetwork):
	def __init__(self, input_size, output_size, name):
		self.name = name

		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope(self.name):
			## 32 8x8 filters, stride=4;   64 4x4 filters, stride=2;   64 3x3, stride 1;   512 fc layer,   output layer
			## 16 8x8 filters, stride=4;   32 4x4 filters, stride=2;   256 fc layer,   output layer

			self.W_conv1 = self.weight_variable([8, 8, 4, 16]) # 32 8x8 filters over 4 channels (frames)
			self.B_conv1 = self.bias_variable([16])
			self.stride1 = 4

			self.W_conv2 = self.weight_variable([4, 4, 16, 32])
			self.B_conv2 = self.bias_variable([32])
			self.stride2 = 2



			# FC layer
			self.W_fc4 = self.weight_variable([9*9*32, 256])#, fanin=11*11*32)
			self.B_fc4 = self.bias_variable([256])#, fanin=11*11*32)

			# FC layer
			self.W_fc5 = self.weight_variable([256, self.output_size])#, fanin=256)
			self.B_fc5 = self.bias_variable([self.output_size])#, fanin=256)


		# Print number of parameters in the network
		self.print_num_of_parameters()



	def __call__(self, input_tensor):
		if type(input_tensor) == list:
			input_tensor = tf.concat(1, input_tensor)

		with tf.variable_scope(self.name):
			# input_tensor is (84, 84, 4)

			#"""
			## NIPS network
			self.h_conv1 = tf.nn.relu( tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, self.stride1, self.stride1, 1], padding='VALID') + self.B_conv1 )
			# max pooling:  self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			self.h_conv2 = tf.nn.relu( tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, self.stride2, self.stride2, 1], padding='VALID') + self.B_conv2 )

			#self.h_conv3 = tf.nn.relu( tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, self.stride3, self.stride3, 1], padding='SAME') + self.B_conv3 )

			self.h_conv2_flat = tf.reshape(self.h_conv2, [-1, 9*9*32])

			self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv2_flat, self.W_fc4) + self.B_fc4)

			self.h_fc5 = tf.identity(tf.matmul(self.h_fc4, self.W_fc5) + self.B_fc5)
			#"""





			"""
			## NIPS w/ Pooling
			self.h_conv1 = tf.nn.relu( tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME') + self.B_conv1 )
			self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, self.stride1, self.stride1, 1], strides=[1, self.stride1, self.stride1, 1], padding='SAME')

			self.h_conv2 = tf.nn.relu( tf.nn.conv2d(self.h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME') + self.B_conv2 )
			self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, self.stride2, self.stride2, 1], strides=[1, self.stride2, self.stride2, 1], padding='SAME')

			self.h_conv2_flat = tf.reshape(self.h_pool2, [-1, 11*11*32])

			self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv2_flat, self.W_fc4) + self.B_fc4)

			self.h_fc5 = tf.identity(tf.matmul(self.h_fc4, self.W_fc5) + self.B_fc5)
			"""



		return self.h_fc5






class QNetworkNature(QNetwork):
	def __init__(self, input_size, output_size, name):
		self.name = name

		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope(self.name):
			## 32 8x8 filters, stride=4;   64 4x4 filters, stride=2;   64 3x3, stride 1;   512 fc layer,   output layer

			self.W_conv1 = self.weight_variable([8, 8, 4, 32]) # 32 8x8 filters over 4 channels (frames)
			self.B_conv1 = self.bias_variable([32])
			self.stride1 = 4

			self.W_conv2 = self.weight_variable([4, 4, 32, 64])
			self.B_conv2 = self.bias_variable([64])
			self.stride2 = 2

			self.W_conv3 = self.weight_variable([3, 3, 64, 64])
			self.B_conv3 = self.bias_variable([64])
			self.stride3 = 1


			# FC layer
			self.W_fc4 = self.weight_variable([7*7*64, 512])#, fanin=11*11*32)
			self.B_fc4 = self.bias_variable([512])#, fanin=11*11*32)

			# FC layer
			self.W_fc5 = self.weight_variable([512, self.output_size])#, fanin=256)
			self.B_fc5 = self.bias_variable([self.output_size])#, fanin=256)


		# Print number of parameters in the network
		self.print_num_of_parameters()





	def __call__(self, input_tensor):
		if type(input_tensor) == list:
			input_tensor = tf.concat(1, input_tensor)

		with tf.variable_scope(self.name):
			# input_tensor is (84, 84, 4)

			self.h_conv1 = tf.nn.relu( tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, self.stride1, self.stride1, 1], padding='VALID') + self.B_conv1 )

			self.h_conv2 = tf.nn.relu( tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, self.stride2, self.stride2, 1], padding='VALID') + self.B_conv2 )

			self.h_conv3 = tf.nn.relu( tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, self.stride3, self.stride3, 1], padding='VALID') + self.B_conv3 )

			self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7*7*64])

			self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4) + self.B_fc4)

			self.h_fc5 = tf.identity(tf.matmul(self.h_fc4, self.W_fc5) + self.B_fc5)


			print self.h_conv1.get_shape()
			print self.h_conv2.get_shape()
			print self.h_conv3.get_shape()


		return self.h_fc5














class DQN(object):
	def __init__(self, state_size,
					   action_size,
					   session,
					   summary_writer = None,
					   exploration_period = 1000,
					   minibatch_size = 32,
					   discount_factor = 0.99,
					   experience_replay_buffer = 10000,
					   target_qnet_update_rate = 0.001,
					   target_qnet_update_frequency = 10000,
					   initial_exploration_epsilon = 1.0,
					   final_exploration_epsilon = 0.05, reward_clipping=-1):


		# Setup the parameters, data structures and networks
		self.state_size = state_size
		self.action_size = action_size

		self.session = session
		self.exploration_period = float(exploration_period)
		self.minibatch_size = minibatch_size
		self.discount_factor = tf.constant(discount_factor)
		self.experience_replay_buffer = experience_replay_buffer
		self.summary_writer = summary_writer

		self.target_qnet_update_rate = target_qnet_update_rate
		self.target_qnet_update_frequency = target_qnet_update_frequency

		self.initial_exploration_epsilon = initial_exploration_epsilon
		self.final_exploration_epsilon = final_exploration_epsilon


		#l1 = 200 #200
		#l2 = 200 #200
		#self.qnet = MLP([self.state_size[0],], [l1, l2, self.action_size], [tf.tanh, tf.tanh, tf.identity], scope="qnet")
		#self.target_qnet = self.qnet.copy(scope="target_qnet")
		#self.qnet_optimizer =tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)#, epsilon=0.01)


		self.qnet = QNetworkNature(self.state_size, self.action_size, "qnet")
		self.target_qnet = QNetworkNature(self.state_size, self.action_size, "target_qnet")

		self.qnet_optimizer =tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.01) ##epsilon=0.01? or 0.0001?


		self.experience_replay = ReplayMemoryFast(self.experience_replay_buffer, self.minibatch_size)

		self.num_training_steps = 0



		# Setup the computational graph
		self.create_graph()



	@staticmethod
	def update_target_network(source_network, target_network, update_rate):
		target_network_update = []
		for v_source, v_target in zip(source_network.variables(), target_network.variables()):
			# this is equivalent to target = (1-alpha) * target + alpha * source
			update_op = v_target.assign_sub(update_rate * (v_target - v_source))
			target_network_update.append(update_op)
		return tf.group(*target_network_update)
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
					qnet_gradients[i] = (tf.clip_by_norm(grad, 5), var)######
					#qnet_gradients[i] = (tf.clip_by_value(grad, -1, 1), var)

			self.qnet_optimize = self.qnet_optimizer.apply_gradients(qnet_gradients)


		with tf.name_scope("target_network_update"):
			self.update_all_targets  = DQN.update_target_network(self.qnet, self.target_qnet, self.target_qnet_update_rate)
			self.hard_copy_to_target = DQN.copy_to_target_network(self.qnet, self.target_qnet)


		self.summarize = tf.merge_all_summaries()




	def store(self, state, action, reward, next_state, is_terminal):
		# next_state should be None in case of terminal states

		# rewards clipping
		#reward = np.clip(reward, -1.0, 1.0)
		reward = np.clip(reward, -2.0, 2.0)

		self.experience_replay.store(state, action, reward, next_state, is_terminal)



	def action(self, state, training = False):
		if self.num_training_steps > self.exploration_period:
			epsilon = self.final_exploration_epsilon
		else:
			epsilon =  self.initial_exploration_epsilon - float(self.num_training_steps) * (self.initial_exploration_epsilon - self.final_exploration_epsilon) / self.exploration_period

		if not training:
			epsilon = 0.05 ##?


		if random.random() <= epsilon:# and not disable_exploration:
			action = random.randint(0, self.action_size-1)
		else:
			action = self.session.run(self.predicted_actions, {self.state:[state] } )[0]

		return action



	def train(self):
		if self.num_training_steps == 0:
			print "Training starts..."
			self.qnet.copy_to(self.target_qnet)


		# sample experience
		minibatch = self.experience_replay.sample()
		if len(minibatch)==0:
			return


		# bach states
		batch_states = np.asarray( [d[0] for d in minibatch] )

		actions = [d[1] for d in minibatch]
		batch_actions = np.zeros( (self.minibatch_size, self.action_size) )
		for i in xrange(self.minibatch_size):
			batch_actions[i, actions[i]] = 1

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


		scores, _, = self.session.run([self.q_values, self.qnet_optimize],
									  { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions} )


		#self.session.run( self.update_all_targets ) ## continuous update

		## hard update (copy) of the weights every # iterations
		if self.num_training_steps % self.target_qnet_update_frequency == 0:
			self.session.run( self.hard_copy_to_target )


		if self.num_training_steps % self.target_qnet_update_frequency == 0:
			#print scores
			print 'mean maxQ in minibatch: ',np.mean(np.max(scores,1))

			str_ = self.session.run(self.summarize, { self.state: batch_states,
										self.next_state: batch_newstates,
										self.next_state_mask: batch_newstates_mask,
										self.rewards: batch_rewards,
										self.action_mask: batch_actions})
			self.summary_writer.add_summary(str_, self.num_training_steps)


		self.num_training_steps += 1





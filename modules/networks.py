"""
Giacomo Spigler
"""
import numpy as np
import random
import tensorflow as tf



class QNetwork(object):
	"""
	Base class for QNetworks. 
	"""
	def __init__(self, input_size, output_size, name):
		self.name = name

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
	"""
	QNetwork used in ``Playing Atari with Deep Reinforcement Learning'', [Mnih et al., 2013].
	It's a Convolutional Neural Network with the following specs:
		L1: 16 8x8 filters with stride 4  +  RELU
		L2: 32 4x4 filters with stride 2  +  RELU
		L3: 256 unit Fully-Connected layer  +  RELU
		L4: [output_size] output units, Fully-Connected
	"""
	def __init__(self, input_size, output_size, name):
		self.name = name

		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope(self.name):
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

			self.h_conv1 = tf.nn.relu( tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, self.stride1, self.stride1, 1], padding='VALID') + self.B_conv1 )
			# max pooling:  self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			self.h_conv2 = tf.nn.relu( tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, self.stride2, self.stride2, 1], padding='VALID') + self.B_conv2 )

			#self.h_conv3 = tf.nn.relu( tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, self.stride3, self.stride3, 1], padding='SAME') + self.B_conv3 )

			self.h_conv2_flat = tf.reshape(self.h_conv2, [-1, 9*9*32])

			self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv2_flat, self.W_fc4) + self.B_fc4)

			self.h_fc5 = tf.identity(tf.matmul(self.h_fc4, self.W_fc5) + self.B_fc5)


		return self.h_fc5






class QNetworkNature(QNetwork):
	"""
	QNetwork used in ``Human-level control through deep reinforcement learning'', [Mnih et al., 2015].
	It's a Convolutional Neural Network with the following specs:
		L1: 32 8x8 filters with stride 4  +  RELU
		L2: 64 4x4 filters with stride 2  +  RELU
		L3: 64 3x3 fitlers with stride 1  +  RELU
		L4: 512 unit Fully-Connected layer  +  RELU
		L5: [output_size] output units, Fully-Connected
	"""
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


		return self.h_fc5











class QNetworkDueling(QNetwork):
	"""
	QNetwork used in ``Human-level control through deep reinforcement learning'', [Mnih et al., 2015].
	It's a Convolutional Neural Network with the following specs:
		L1: 32 8x8 filters with stride 4  +  RELU
		L2: 64 4x4 filters with stride 2  +  RELU
		L3: 64 3x3 fitlers with stride 1  +  RELU
		L4a: 512 unit Fully-Connected layer  +  RELU
		L4b: 512 unit Fully-Connected layer  +  RELU
		L5a: 1 unit FC + RELU (State Value)
		L5b: #actions FC + RELU (Advantage Value)
		L6: Aggregate V(s)+A(s,a)
	"""
	def __init__(self, input_size, output_size, name):
		self.name = name

		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope(self.name):
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
			self.W_fc4a = self.weight_variable([7*7*64, 512])#, fanin=11*11*32)
			self.B_fc4a = self.bias_variable([512])#, fanin=11*11*32)

			self.W_fc4b = self.weight_variable([7*7*64, 512])#, fanin=11*11*32)
			self.B_fc4b = self.bias_variable([512])#, fanin=11*11*32)

			# FC layer
			self.W_fc5a = self.weight_variable([512, 1])#, fanin=256)
			self.B_fc5a = self.bias_variable([1])#, fanin=256)

			self.W_fc5b = self.weight_variable([512, self.output_size])#, fanin=256)
			self.B_fc5b = self.bias_variable([self.output_size])#, fanin=256)



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

			self.h_fc4a = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4a) + self.B_fc4a)
			self.h_fc4b = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4b) + self.B_fc4b)

			self.h_fc5a_value     = tf.identity(tf.matmul(self.h_fc4a, self.W_fc5a) + self.B_fc5a)
			self.h_fc5b_advantage = tf.identity(tf.matmul(self.h_fc4b, self.W_fc5b) + self.B_fc5b)

			self.h_fc6 = self.h_fc5a_value + ( self.h_fc5b_advantage - tf.reduce_mean(self.h_fc5b_advantage, reduction_indices=[1,], keep_dims=True) )


		return self.h_fc6








class ActorCritic_MLP(QNetwork):
	def __init__(self, input_size, output_size, actor_or_critic, name):
		"""
		actor_or_critic=0 for actor, 1 for critic. The only difference is in the output transfer function (tanh for actor, identity for critic)
		"""
		self.name = name
		self.actor_or_critic = actor_or_critic

		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope(self.name):
			l1 = 200
			l2 = 200

			self.W_fc1 = self.weight_variable([self.input_size, l1])
			self.B_fc1 = self.bias_variable([l1])

			self.W_fc2 = self.weight_variable([l1, l2])
			self.B_fc2 = self.bias_variable([l2])

			self.W_out = self.weight_variable([l2, self.output_size], uniform=0.003)
			self.B_out = self.bias_variable([self.output_size], uniform=0.003)


		# Print number of parameters in the network
		self.print_num_of_parameters()


	def __call__(self, input_tensor):
		if type(input_tensor) == list:
			input_tensor = tf.concat(1, input_tensor)

		with tf.variable_scope(self.name):
			self.h_fc1 = tf.nn.relu(tf.matmul(input_tensor, self.W_fc1) + self.B_fc1)
			self.h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, self.W_fc2) + self.B_fc2)

			if self.actor_or_critic==0: # ACTOR
				self.out = tf.nn.tanh(tf.matmul(self.h_fc2, self.W_out) + self.B_out)
			else: # CRITIC
				self.out = tf.identity(tf.matmul(self.h_fc2, self.W_out) + self.B_out)

		return self.out






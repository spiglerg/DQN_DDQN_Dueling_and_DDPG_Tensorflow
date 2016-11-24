import gym
import time
import logging
import shutil
import os
import sys

import tensorflow as tf

import gc
gc.enable()



from modules.ddpg import *
from modules.env_utils import *


# Save video rendering every Xth game played
def multiples_video_schedule(episode_id):
	return episode_id % 100 == 0 # and episode_id>0
	#return episode_id % 200 == 0


print
print "Usage:"
print "  ",sys.argv[0]," [optional: path_to_ckpt_file] [optional: True/False test mode]"
print
print




outdir = "gym_results"



ENV_NAME = 'Pendulum-v0' # BipedalWalker-v2
TOTAL_FRAMES = 200000 ## TRAIN
MAX_TRAINING_STEPS = 500 ## MAX STEPS BEFORE RESETTING THE ENVIRONMENT
TESTING_GAMES = 100 # no. of games to average on during testing
MAX_TESTING_STEPS = 500 #5 minutes  '/3' because gym repeating the last action 3-4 times already!
TRAIN_AFTER_FRAMES = 1000
epoch_size = 5000 # every how many frames to test

"""
ENV_NAME = 'BipedalWalker-v2'
TOTAL_FRAMES = 20000000 ## TRAIN
MAX_TRAINING_STEPS = 2000 ## MAX STEPS BEFORE RESETTING THE ENVIRONMENT
TESTING_GAMES = 100 # no. of games to average on during testing
MAX_TESTING_STEPS = 2000 #5 minutes  '/3' because gym repeating the last action 3-4 times already!
TRAIN_AFTER_FRAMES = 50000
epoch_size = 50000 # every how many frames to test
"""


MAX_NOOP_START = 0


LOG_DIR = outdir+'/'+ENV_NAME+'/logs/'
if os.path.isdir(LOG_DIR):
	shutil.rmtree(LOG_DIR)
journalist = tf.train.SummaryWriter(LOG_DIR)



# Build environment
env = gym.make(ENV_NAME)


# Initialize Tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.InteractiveSession(config=config)


# Create DQN agent
agent = DDPG(state_size=env.observation_space.shape,
			action_size=env.action_space.shape[0],
			session=session,
			summary_writer = journalist,
			exploration_period = 20000,
			minibatch_size = 64,
			discount_factor = 0.97,
			experience_replay_buffer = 500000,
			target_qnet_update_frequency = 1000,
			initial_exploration_epsilon = 0.2,
			final_exploration_epsilon = 0.2)#,
#			reward_clipping = 1.0)



session.run(tf.initialize_all_variables())
journalist.add_graph(session.graph)

saver = tf.train.Saver(tf.all_variables())

env.monitor.start(outdir+'/'+ENV_NAME,force = True, video_callable=multiples_video_schedule)
logger = logging.getLogger()
logging.disable(logging.INFO)


# If an argument is supplied, load the specific checkpoint.
test_mode = False
if len(sys.argv)>=2:
	print sys.argv[1]
	saver.restore(session, sys.argv[1])
if len(sys.argv)==3:
	test_mode = bool(sys.argv[2])



num_frames = 0
num_games = 0

current_game_frames = 0


last_time = time.time()
last_frame_count = 0.0


state = env.reset()
agent.noise.reset()
while num_frames <= TOTAL_FRAMES+1:
	if test_mode:
		env.render()

	num_frames += 1
	current_game_frames += 1


	# Pick action given current state
	if not test_mode:
		action = agent.action(state, training = True)
	else:
		action = agent.action(state, training = False)

	if current_game_frames < MAX_NOOP_START:
		action = 0

	# Perform the selected action on the environment
	next_state,reward,done,_ = env.step(action)


	# Store experience
	if current_game_frames >= MAX_NOOP_START:
		agent.store(state,action,reward,next_state,done)
	state = next_state


	# Train agent
	if num_frames>=TRAIN_AFTER_FRAMES:
		agent.train()

	if done or current_game_frames > MAX_TRAINING_STEPS:
		state = env.reset()
		agent.noise.reset()
		current_game_frames = 0
		num_games += 1


	# Print an update
	if num_frames % epoch_size == 0:
		new_time = time.time()
		diff = new_time - last_time
		last_time = new_time

		elapsed_frames = num_frames - last_frame_count
		last_frame_count = num_frames

		print "frames: ",num_frames,"    games: ",num_games,"    speed: ",(elapsed_frames/diff)," frames/second"


	# Save the network's parameters after every epoch
	if num_frames % epoch_size == 0  and  num_frames > TRAIN_AFTER_FRAMES:
		saver.save(session, outdir+"/"+ENV_NAME+"/model_"+str(num_frames/1000)+"k.ckpt")

		print
		print "epoch:  frames=",num_frames,"   games=",num_games


	## Testing -- it's kind of slow, so we're only going to test every 2 epochs
	if num_frames % (2*epoch_size) == 0  and num_frames > TRAIN_AFTER_FRAMES:
		total_reward = 0
		avg_steps = 0
		for i in xrange(TESTING_GAMES):
			state = env.reset()
			agent.noise.reset()
			frm = 0
			while frm < MAX_TESTING_STEPS:
				frm += 1
				#env.render()
				action = agent.action(state, training = False) # direct action for test
				state,reward,done,_ = env.step(action)

				total_reward += reward
				if done:
					break

			avg_steps += frm
		avg_reward = float(total_reward)/TESTING_GAMES

		str_ = session.run( tf.scalar_summary('test reward ('+str(epoch_size/1000)+'k)', avg_reward) )
		journalist.add_summary(str_, num_frames) #np.round(num_frames/epoch_size)) # in no. of epochs, as in Mnih

		print '  --> EVALUATION AVERAGE REWARD: ',avg_reward,'   avg steps: ',(avg_steps/TESTING_GAMES)

		state = env.reset()
		agent.noise.reset()


env.monitor.close()
journalist.close()


## Save the final network
saver.save(session, outdir+"/"+ENV_NAME+"/final.ckpt")




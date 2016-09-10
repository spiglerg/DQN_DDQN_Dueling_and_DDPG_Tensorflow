# Atari_DQN_Nature_TF_Implementation
Tensorflow + OpenAI Gym implementation of the Deep Q-Network agent described in ``Human-level control through deep reinforcement learning'', [Mnih et al., 2015].

The current code has been tested on Seaquest, but it should work with any other Atari environment.



Usage:
* Train:
$ python gym_dqn_atari.py -- trains the chosen game (customizable by modifying the code) for 10 million frames.

* Test:
$ python gym_dqn_atari.py [path to trained model file] True

* Pretrained test:
$ python gym_dqn_atari.py pretrained/Seaquest-10M.ckpt True



Apart from the implementation per se, you might be interested in the implementation of the Experience Replay Memory, which is pretty fast, and the wrapper to pre-process OpenAI Gym frames for Atari games.

I have tried to keep the implementation as simple and minimal as possible, but still retaining all the important functionalities.
For anything else, feel free to contact me!



![alt tag](images/plot_seaquest_10M.png)


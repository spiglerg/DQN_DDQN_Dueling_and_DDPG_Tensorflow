# DQN_and_DDPG_Tensorflow
Tensorflow + OpenAI Gym implementation of two popular Deep Reinforcement Learning models:
* **Deep Q-Network (DQN)**, as described in ``Human-level control through deep reinforcement learning'', [Mnih et al., 2015].
* **Deep Deterministic Policy Gradient (DDPG)**, as described in ``Continuous control with deep reinforcement learning'', [Lillicrap et. al, 2015].



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


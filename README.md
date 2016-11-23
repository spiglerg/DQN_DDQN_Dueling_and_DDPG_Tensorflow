# Tensorflow implementation of [Dueling] [D]DQN + DDPG  Deep Reinforcement Learning Algorithms
Tensorflow + OpenAI Gym implementation of two popular Deep Reinforcement Learning models:
* **Deep Q-Network** (DQN), as described in ``Human-level control through deep reinforcement learning'', [Mnih et al., 2015] (both Nature and NIPS networks available).
* **Double Deep Q-Network** (DDQN), as described in ``Deep Reinforcement Learning with Double Q-Learning'', [van HAsselt et al., 2015]  (this can be selected by setting `DoubleDQN=True')
* **Dueling Network Architecture**, as described in ``Dueling Network Architectures for Deep Reinforcement Learning'', [Wang et al., 2016]. The network can be selected by changing `qnet' and `target_qnet' in modules/dqn.py
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


(Older) DQN results:
![alt tag](images/plot_seaquest_DQN_10M.png)


import gym
import MaxRewardsGame
import numpy as np
from gym import wrappers
from dqn import DeepQNetwork
import pdb

max_episodes = 10000
training_max_step = 10000
playing_max_step = 30

def train(env, dqn):
	print("-------------------- Train start -------------------")
	env.enabled = False
	for episode in range(max_episodes):
		env.reset()
		step = 0
		cur_state = env.pos2state(env.cur_pos)
		coin_idx = env.coins_permutation_index()
		observation = np.array((coin_idx, cur_state))
		for i in range(training_max_step):
		# while True:
			# pdb.set_trace()
			action = dqn.choose_action(observation)
			observation_, reward, isDone, info = env.step(action)
			dqn.update_memory(observation, action, reward, observation_)
			if step > 20 and step % 5 == 0:
				dqn.learn(episode)
			observation = observation_
			if isDone:
				break
			step += 1
		dqn.reward_his.append(env.total_reward)
		dqn.reduce_epislon()
		# print("cur_episode = ", episode, " reward = ", env.total_reward)
	dqn.saver.save(dqn.sess,'ckpt/MRG.ckpt')

def play(env, dqn):
	# dqn.load_model()
	env.enabled = True
	env.reset()
	cur_state = env.pos2state(env.cur_pos)
	coin_idx = env.coins_permutation_index()
	observation = np.array((coin_idx, cur_state))
	print("--------------- Play start ----------------")
	print("init_state = ", cur_state, "init_coin_idx = ", coin_idx)
	step = 0
	for i in range(playing_max_step):
		action = dqn.greedy_choose_action(observation)
		observation_, reward, isDone, info = env.step(action)
		observation = observation_
		if isDone:
			break
		step += 1
	print("--------------- Params ---------------")
	#dqn.output_params()


if __name__ == "__main__":
	env = gym.make('MaxRewardsGame-v0')
	env = wrappers.Monitor(env, "./gym-results", force=True)
	dqn = DeepQNetwork(
		env = env,
		n_actions = len(env.action),
		n_features = 2,
		learning_rate = 0.001,
		discount_rate = 0.9,
		e_greedy_max = 1,
		replace_target_param_interval=5000,
		memory_size=20000,
		batch_size=1024,
		e_greedy_decrement=1/(max_episodes*8),
		output_graph=False)
	train(env, dqn)
	play(env, dqn)
	env.close()
	dqn.plot_cost()
	dqn.plot_reward()

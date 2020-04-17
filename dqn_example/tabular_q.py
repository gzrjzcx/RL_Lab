import gym
import MaxRewardsGame
import numpy as np
from gym import wrappers

env = gym.make('MaxRewardsGame-v0')
env = wrappers.Monitor(env, "./gym-results", force=True)

class Agent:
	def __init__(self, env):

		self.l_r = 1
		self.epsilon = 0.8
		self.discount = 0.99995
		self.env = env
		self.coin_permutation = 1 << len(self.env.coins)
		self.valid_action = dict()
		self._initQMartrix()
	
	def _initQMartrix(self):
		self.q_martrix = np.zeros([self.coin_permutation,
									self.env.state_size,
									len(self.env.action)])

	def _setValidActionDict(self):
		v_a = dict()
		for action in env.action:
			if env.checkActionIsValid(env.cur_pos, action):
				v_a.update({action : 1})
			else:
				v_a.update({action : 0})
		return v_a		

	def _getValidActionList(self):
		v_a = []
		for action in env.action:
			if env.checkActionIsValid(env.cur_pos, action):
				v_a.append(action)
		return v_a

	def greedyActionSample(self, cur_state):
		v_a = self._setValidActionDict()
		max_q = float("-inf")
		argmax_a = None
		for action, isValid in v_a.items():
			coin_idx = env.coins_permutation_index()
			action_idx = env.action2index(action)
			if isValid == 1 and self.q_martrix[coin_idx, cur_state, action_idx] > max_q:
				max_q = self.q_martrix[coin_idx, cur_state, action_idx]
				argmax_a = action
		if argmax_a == None:
			raise LookupError('eGreedyActionSample cannot pick up the best q action')
		else:
			return argmax_a

	def eGreedyActionSample(self, cur_state):
		if np.random.sample() < self.epsilon:
			v_a = self._getValidActionList()
			action = (np.random.choice(v_a, 1))[0]
			return np.str(action)
		else:
			v_a = self._setValidActionDict()
			max_q = float("-inf")
			argmax_a = None
			coin_idx = env.coins_permutation_index()
			for action, isValid in v_a.items():
				action_idx = env.action2index(action)
				if isValid == 1 and self.q_martrix[coin_idx, cur_state, action_idx] > max_q:
					max_q = self.q_martrix[coin_idx, cur_state, action_idx]
					argmax_a = action
			if argmax_a == None:
				raise LookupError('eGreedyActionSample cannot pick up the best q action')
			else:
				return argmax_a

	def train(self, max_episodes, max_step):
		env.enabled = False # disable video monitoring
		for e in range(max_episodes):
			if e % 1000 == 0:
				print("--------------- Train Start, e = ", e, " ---------------")
			env.reset()
			cur_state = env.pos2state(env.cur_pos)
			for i in range(max_step):
				action = self.eGreedyActionSample(cur_state)
				coin_idx = env.coins_permutation_index()
				next_state, r, isDone, info = env.step(action)
				action_idx = env.action2index(action)
				self.q_martrix[coin_idx, cur_state, action_idx] += self.l_r * (
					r + self.discount * np.amax(self.q_martrix[coin_idx, next_state, :]) - 
					self.q_martrix[coin_idx, cur_state, action_idx])
				# print(coin_idx, cur_state, "->", next_state, action_idx)
				# print(self.q_martrix[coin_idx, cur_state, action_idx], self.q_martrix[coin_idx, next_state, action_idx])
				cur_state = next_state
				if isDone:
					break

	def play(self):
		print("-----------  Play Start --------- ")
		env.reset()
		env.enabled = True
		cur_state = 0
		max_step = 100
		for _ in range(max_step):
			action = self.greedyActionSample(cur_state)
			next_state, r, isDone, info = env.step(action)
			print(cur_state, action, next_state, end=" ->\n")
			cur_state = next_state
			if isDone is True:
				break
		# print(self.q_martrix[7, :, :])
		self.output_file()

	def output_file(self):
		f = open("./gym-results/q.txt", "w+")
		f.write("{0}\t|\t{1}\t|\t{2}\t|\t{3}\t|\t{4}\n".format("state", "left", "right", "up", "down"))
		coin_idx = 7
		while coin_idx >= 0:
			f.write("------------------ coin_idx = {0} ----------------\n".format(coin_idx))
			for s in range(env.state_size):
				f.write(" {0}\t|\t{1:.3f}\t|\t{2:.3f}\t|\t{3:.3f}\t|\t{4:.3f}\n".format(s, 
															self.q_martrix[coin_idx, s, 0],
															self.q_martrix[coin_idx, s, 1],
															self.q_martrix[coin_idx, s, 2],
															self.q_martrix[coin_idx, s, 3] ))
			coin_idx -= 1
		f.close()

env.reset()
agent = Agent(env)
agent.train(1000, 200)
agent.play()

env.close()

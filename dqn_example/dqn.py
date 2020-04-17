import tensorflow as tf
import numpy as np
import pdb

class DeepQNetwork:

	def __init__(
		self,
		env,
		n_actions,
		n_features,
		learning_rate=0.01,
		discount_rate=0.9,
		e_greedy_max=0.8,
		e_greedy_min=0.4,
		replace_target_param_interval=300,
		memory_size=200,
		batch_size=32,
		e_greedy_decrement=None,
		output_graph=False,
	):
		self.env = env
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = discount_rate
		self.replace_target_param_interval = replace_target_param_interval
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.e_greedy_decrement = e_greedy_decrement
		self.epsilon_min = e_greedy_min
		self.epsilon = e_greedy_max
		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size, n_features*2+2))

		self.sess = tf.Session()
		if output_graph:
			tf.summary.FileWriter("logs/", self.sess.graph)
	
		# pdb.set_trace()
		self._build_net()
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t,e) for t, e in zip(t_params, e_params)]		

		self.sess.run(tf.global_variables_initializer())
		# obs = np.array((7,0))
		# obs = obs[np.newaxis, :]
		# self.sess.run(self.q_eval, feed_dict={self.s: obs})
		self.cost_his = []
		self.reward_his = []

		self.saver = tf.train.Saver(max_to_keep=1)
		self.output_params()

	def _build_net(self):
		# build evaluate network
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
		with tf.variable_scope('eval_net'):
			# c_names指的是tf中的collection，用于收集神经网络用到的参数weight，即eval_net_params变量代表的参数
			# w_initializer建立weight的随机参数，b_initializer建立bias的随机参数
			c_names, n_l1, w_initializer, b_initializer = \
				['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
				tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=b_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_eval = tf.matmul(l1, w2) + b2

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		# build target network
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
		with tf.variable_scope('target_net'):
			c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
				self.q_next = tf.matmul(l1, w2) + b2

	def output_params(self):
		vars = tf.trainable_variables()
		print(vars) #some infos about variables...
		vars_vals = self.sess.run(vars)
		for var, val in zip(vars, vars_vals):
			print("var: {}, value: {}".format(var.name, val))

	def load_model(self):
		self.saver.restore(self.sess, 'ckpt/MRG.ckpt')

	def _filter_invalid_action(self, actions_values, observation):
		valid_actions = self.env.getValidActionDict(observation)
		min_q = float("-inf")
		for action, isValid in valid_actions.items():
			if isValid is not 1:
				actions_values[action] = min_q
		return actions_values

	def choose_action(self, observation):
		cur_pos = self.env.state2pos(observation[1])
		observation = observation[np.newaxis, :]
		if np.random.uniform() < self.epsilon:
			v_a = self.env.getValidActionList(cur_pos)
			action = (np.random.choice(v_a, 1))[0]
		else:
			actions_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			filtered_action_values = self._filter_invalid_action(np.squeeze(actions_values), cur_pos)
			action = np.argmax(filtered_action_values)
		return action

	def greedy_choose_action(self, observation):
		cur_pos = self.env.state2pos(observation[1])
		coin_idx = observation[0]
		cur_state = observation[1]
		observation = observation[np.newaxis, :]
		actions_values = self.sess.run(self.q_eval, feed_dict={self.s: observation})
		print("coin_idx = ", coin_idx, " cur_state = ", cur_state, " | actions_values = ", actions_values)
		filtered_action_values = self._filter_invalid_action(np.squeeze(actions_values), cur_pos)
		action = np.argmax(filtered_action_values)
		return action

	def update_memory(self, obs, a, r, obs_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0

		experience = np.hstack((obs, [a, r], obs_))
		# print("experience = ", experience)
		idx = self.memory_counter % self.memory_size
		self.memory[idx, :] = experience
		self.memory_counter += 1		

	def learn(self, episode):
		if self.learn_step_counter % self.replace_target_param_interval == 0:
			self.sess.run(self.replace_target_op)
			print("replaced target network parameters susccessfully, episode = ", episode,
				" epsilon = ", self.epsilon)

		if self.memory_counter > self.memory_size:
			sample_idx = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_idx = np.random.choice(self.memory_counter, size=self.batch_size)

		# get all single experience from memory buffer depends on the smapled index
		memory_batch = self.memory[sample_idx, :]

		q_next, q_eval = self.sess.run(
			[self.q_next, self.q_eval],
			feed_dict = {
				self.s_: memory_batch[:, -self.n_features:],
				self.s: memory_batch[:, :self.n_features  ]
			})

		q_target = q_eval.copy()
		batch_idx = np.arange(self.batch_size, dtype=np.int32)
		eval_action_idx = memory_batch[:, self.n_features].astype(int)
		reward = memory_batch[:, self.n_features + 1]
		# print("q_next = ", q_next)
		# print("q_eval = ", q_eval)
		# print("eval_action_idx = ", eval_action_idx)
		eval_state_idx = memory_batch[:, self.n_features-1].astype(int)
		# print("eval_state_idx = ", eval_state_idx)
		for i in batch_idx:
			# print(q_next[i])
			self._filter_invalid_action(q_next[i], self.env.state2pos(eval_state_idx[i]))
		# print("q_next = ", q_next)
		q_target[batch_idx, eval_action_idx] = reward + self.gamma * np.max(q_next, axis=1)

		# training
		_, self.cost = self.sess.run([self._train_op, self.loss],
									 feed_dict = {self.s: memory_batch[:, :self.n_features],
									 			  self.q_target: q_target})
		# tf.summary.scalar('loss', self.cost)
		# tf.summary.scalar('cumulative_reward', self.env.total_reward)
		# tf.summary.FileWriter("logs/", self.sess.graph)

		self.cost_his.append(self.cost)
		# if self.epsilon > self.epsilon_min:
		# 	self.epsilon -= self.e_greedy_decrement 
		self.learn_step_counter += 1

	def reduce_epislon(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.e_greedy_decrement

	def plot_cost(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.savefig('gym-results/dqn.pdf')
		plt.clf()

	def plot_reward(self):
		import matplotlib.pyplot as plt
		# pdb.set_trace()
		r = np.array(self.reward_his)
		y = np.mean(r[:len(self.reward_his)//100*100].reshape(-1, 100), axis=0)
		plt.plot(y)
		plt.ylabel('cumulative_reward')
		plt.xlabel('episodes')
		plt.savefig('gym-results/cumulative_reward.pdf')
		plt.clf()
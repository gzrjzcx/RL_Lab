import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient(object):
	"""docstring for PolicyGradient"""
	def __init__(
		self,
		n_actions,
		n_features,
		learning_rate,
		reward_decay,
		output_graph=False
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []

		self._build_net()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		with tf.name_scope('inputs'):
			self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observations')
			self.tf_actions = tf.placeholder(tf.int32, [None, ], name='actions_num')
			self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value')

		l1 = tf.layers.dense(
			inputs=self.tf_obs,
			units=10,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc_1')

		logits = tf.layers.dense(
			inputs=l1,
			units=self.n_actions,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc_2')

		self.actions_prob = tf.nn.softmax(logits, name='actions_prob')

		with tf.name_scope('loss'):
			'''
			policy gradient实际上是没有误差的，但是会通过反向传递来改变policy，使某些动作选取的概率更大，
			所以我们可以把它当作是误差的反向传递，即loss。传递的实际上就是log(policy(s, a))*v，v就是value-function,
			用来告诉这次梯度应该向着何种方向更新。
			'''
			# neg_log_prob其实就是当前policy下选择每个动作的negative log probability
			neg_log_prob = tf.reduce_sum(-tf.log(self.actions_prob)*tf.one_hot(self.tf_actions, self.n_actions), axis=1)
			loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def choose_action(self, obs):
		prob_weights = self.sess.run(self.actions_prob, feed_dict={self.tf_obs: obs[np.newaxis, :]})
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
		return action

	def store_transition(self, s, a, r):
		self.ep_obs.append(s)
		self.ep_actions.append(a)
		self.ep_rewards.append(r)

	def learn(self):
		discounted_ep_r_norm = self._discount_and_norm_rewards()
		print("discounted_ep_r_norm = ")
		print(discounted_ep_r_norm)

		self.sess.run(self.train_op, feed_dict={
			self.tf_obs: np.vstack(self.ep_obs),
			self.tf_actions: np.array((self.ep_actions)),
			self.tf_vt: discounted_ep_r_norm
		})

		self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []
		return discounted_ep_r_norm


	'''
	这里实际上使用的还是total reward作为policy gradient的V值，来指导梯度的更新，但是对reward做了折扣和归一化处理
	也可以使用其他的形式，比如advantage function(Q-V)，甚至GAE
	'''
	def _discount_and_norm_rewards(self):
		discounted_ep_rewards = np.zeros_like(self.ep_rewards)
		running_add = 0
		for t in reversed(range(0, len(self.ep_rewards))):
			# 越靠近当前timestep的动作，影响越大，折扣越少
			running_add = running_add * self.gamma + self.ep_rewards[t]
			print("running_add = ", running_add)
			discounted_ep_rewards[t] = running_add

		print("original running_add = ", discounted_ep_rewards)
		discounted_ep_rewards -= np.mean(discounted_ep_rewards)
		discounted_ep_rewards /= np.std(discounted_ep_rewards)
		return discounted_ep_rewards



















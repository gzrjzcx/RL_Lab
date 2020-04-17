import numpy as np
import tensorflow as tf

class PolicyGradient(object):
	"""docstring for PolicyGradient"""
	def __init__(
		self,
		env,
		n_actions,
		n_features,
		learning_rate,
		discount_rate,
		output_graph=False
	):
		self.env = env
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = discount_rate

	def _build_net(self):
		with tf.name_scope('inputs'):
			self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name='observations')
			self.tf_actions = tf.placeholder(tf.float32, [None, ], name='actions_num')
			self.tf_vt = tf.placeholder(tf.float32, [None, ], name='actions_value')

		l1 = tf.layers.dense(
			inputs=self.tf_obs,
			units=10,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc_1')

		actions_output = tf.layers.dense(
			inputs=l1,
			units=self.n_actions,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name='fc_2')

		self.actions_prob = tf.nn.softmax(actions_output, name='actions_prob')

		with tf.name_scope('loss'):
			
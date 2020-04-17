import gym
GAMMA = 0.99
MAX_STEP = 100000000
env = gym.make('CartPole-v0')
env.reset()

import tensorflow as tf
import numpy as np

STATE_SIZE = 4
ACTION_SIZE = 2

def InitQNet(input_state, scope, trainable):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        hidden = tf.layers.dense(input_state, 32, trainable = trainable, activation = tf.nn.relu,
                                 kernel_initializer = tf.initializers.orthogonal())
        output = tf.layers.dense(hidden, ACTION_SIZE, trainable = trainable, activation = tf.nn.relu,
                                 kernel_initializer = tf.initializers.orthogonal())
        return output

def EpsilonGreedy(q_output, epsilon):
    if np.random.rand()> epsilon:
        return np.argmax(q_output)
    else:
        return np.random.randint(ACTION_SIZE)
    
with tf.Session() as sess:
    input_test = tf.placeholder(tf.float32, [None, STATE_SIZE], name = 'input_test')
    output_test = InitQNet(input_test, 'test',True)
    sess.run(tf.global_variables_initializer())
    
    #跑一次前向试试
    output_test_result = sess.run(output_test, 
                                feed_dict = {
                                    input_test: np.array([1.0,0.9,0.8,0.7]).reshape(-1,STATE_SIZE),
                                })
    print('Output Result：', output_test_result)
    print('Chosen Action:', EpsilonGreedy(output_test_result, 0.5))

tf.reset_default_graph()
with tf.Session() as sess:
    #初始化网络结构
    online_input_ph = tf.placeholder(tf.float32, [None, STATE_SIZE], name = 'online_input')
    target_input_ph = tf.placeholder(tf.float32, [None, STATE_SIZE], name = 'target_input')
    online_Q = InitQNet(online_input_ph, 'online',True)
    target_Q = InitQNet(target_input_ph, 'target',False)
    
    #定义损失函数
    action_ph = tf.placeholder(tf.int32, [None, ], name = 'action_ph')
    a_indices = tf.stack([tf.range(tf.shape(action_ph)[0], dtype=tf.int32), action_ph], axis=1)
    y_ph = tf.placeholder(tf.float32, [None, ], name = 'y_ph') # r+Q^target
    loss = tf.square(y_ph - tf.gather_nd(params = online_Q, indices = a_indices))
    
    #定义optimizer
    learning_rate = 1e-6
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    #初始化参数
    sess.run(tf.global_variables_initializer())
    
    #online target网络进行同步
    params_online = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='online')
    params_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
    update_target_net_op = [tf.assign(o,n) for o,n in zip(params_online, params_target)]    
    
    last_action = 0
    last_reward = .0
    last_state = np.array([.0, .0, .0, .0])
    for i in range(MAX_STEP):
        if i%10000 == 0:
            sess.run(update_target_net_op)
        
        #选取当前步骤要选择的动作
        online_Q_result = sess.run(online_Q, 
                            feed_dict = {
                                online_input_ph: last_state.reshape(-1,STATE_SIZE),
                            })
        epsilon = 1-float(i)/MAX_STEP 
        chosen_action = EpsilonGreedy(output_test_result, epsilon)
        
        #游戏环境往前走一步
        new_state, new_reward, is_terminal, _ = env.step(chosen_action)
        
        #神经网络先算y（r+Q^target）
        target_Q_result = sess.run(target_Q, 
                            feed_dict = {
                                target_input_ph: new_state.reshape(-1,STATE_SIZE),
                            })
        y = last_reward + GAMMA * np.max(target_Q_result)
        #再计算optimizer
        sess.run(optimizer, feed_dict = {
                                y_ph: y.reshape(-1,),
                                action_ph: np.array(last_action).reshape(-1,),
                                online_input_ph: last_state.reshape(-1,STATE_SIZE),
                            })
        
        last_action = chosen_action
        last_reward = new_reward
        last_state = new_state
        
        if is_terminal:
            env.reset()
            last_action = 0
            last_reward = .0
            last_state = np.array([.0, .0, .0, .0])
            print("~~~~Train End~~~~")

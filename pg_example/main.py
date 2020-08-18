import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from pg import PolicyGradient

DISPLAY_REWARD_THRESHOLD = -2000
RENDER = False

env = gym.make('MountainCar-v0')
env.seed(1)
pg = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.02,
                    reward_decay=0.995)

"""
这里先用env.unwrapper来移除env上包裹的所有wrapper（其中一个让env的每个episode最大step设置为200），
然后给env包裹上Monitor Wrapper来实现渲染
"""
env = env.unwrapped
env = wrappers.Monitor(env, "../gym-results", force=True)


for episode in range(1000):
    obs = env.reset()
    while True:
        if RENDER:
            env.render()

        action = pg.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        # print("cur_reward = ", reward)
        pg.store_transition(obs, action, reward)

        if done:
            ep_rewards_sum = sum(pg.ep_rewards)
            if 'running_reward' not in globals():
                running_reward = ep_rewards_sum
            else:
                running_reward = running_reward * 0.99 + ep_rewards_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print("episode: ", episode, "   reward: ", running_reward)
            vt = pg.learn()

            # if episode == 30:
            #     plt.plot(vt)
            #     plt.xlabel("episode")
            #     plt.ylabel("normalized state action value")
            #     plt.save()
            break

        obs = obs_

env.close()


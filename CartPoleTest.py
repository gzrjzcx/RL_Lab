import gym
import time
from gym import wrappers

env = gym.make('CartPole-v0')
env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    time.sleep(0.1)
    if done:
        break
env.close()

print('State(游戏状态):',env.observation_space, state)
print('Action(动作)：', env.action_space, action)
print('Reward(奖励)：', reward)
print('Is_Done:(游戏是否结束)：', done)

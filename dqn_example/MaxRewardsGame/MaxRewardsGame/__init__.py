from gym.envs.registration import register

register(
	id='MaxRewardsGame-v0',
	entry_point = 'MaxRewardsGame.envs:MAXREWARDSGAME')
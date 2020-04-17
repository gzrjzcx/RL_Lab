import numpy as np
from gym import error, spaces, utils, wrappers
from gym.utils import seeding
from termcolor import colored, cprint

'''
'coin_idx' is used to specify the current coins case. 
E.g.,  assuming the first coin is consumed after one step,
then the q-table will change to a new empty list, and we should update
the following q-value for the new q-table from now on,
until the agent encountering and consuming the next coin.

Observation: [cur_state, coin_idx]  # 2
Action: ["left", "right", "up", "down"]  # 4
'''

class MAXREWARDSGAME(gym.Env):
	"""docstring for MAXREWARDSGAME"""
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 2
	}
	def __init__(self):
		self.map_width = 10
		self.map_height = 10
		self.state_size = self.map_height * self.map_width
		self.action = [0, 1, 2, 3] # left, right, up, down
		self.state = np.arange(self.state_size)
		self.done = False
		self.reward = 0
		self.r_matrix = np.zeros(100)
		self.coins = dict() # in state ascending order
		self.coins[17] = 0.2
		# self.coins[51] = 0.5
		# self.coins[88] = 0.9
		self.reward_per_step = -1/10000
		self.start_pos = [0, 0]
		self.end_pos = [9, 9]
		self.cur_pos = self.start_pos[:]
		self.total_reward = 0
		# self.path = [(0, 1), (1,1), (2,1), (3,1), (4,3), (14,3), (24,3), (34,1), (35,3), (45,3)]
		self._init_r_matrix()
		self.viewer = None
		self._coin_geoms = dict() #_coin_geoms[state] = [circle_geom_obj, coinTrans]

	def _init_r_matrix(self):
		for i in range(self.state_size):
			for key, value in self.coins.items():
				if i == key:
					self.r_matrix[i] = value
					break
			if i == self.pos2state(self.end_pos):
				self.r_matrix[i] = 0.8
			# print("r_matrix[{0}] = {1}".format(i, self.r_matrix[i]))

	def pos2state(self, pos):
		return pos[1] * self.map_width + pos[0]

	def state2pos(self, state):
		return (state % self.map_width, (state - state % self.map_width) / self.map_height )

	def checkActionIsValid(self, pos, action):
		if pos[0] == 0 and action == 0:
			return False
		if pos[0] == self.map_width - 1 and action == 1:
			return False
		if pos[1] == 0 and action == 2:
			return False
		if pos[1] == self.map_height - 1 and action == 3:
			return False
		return True 

	def getRewardAndRemoveCoin(self, pos):
		state = self.pos2state(pos)
		r = self.reward_per_step
		if self.r_matrix[state] > 0:
			# print("remove coin::::  state = ", state)
			r += self.r_matrix[state]
			self.r_matrix[state] = 0
			if state in self._coin_geoms:
				self.remove_coin()
		return r

	def getValidActionDict(self, cur_pos):
		v_a = dict()
		for action in self.action:
			if self.checkActionIsValid(cur_pos, action):
				v_a.update({action : 1})
			else:
				v_a.update({action : 0})
		return v_a		

	def getValidActionList(self, cur_pos):
		v_a = []
		for action in self.action:
			if self.checkActionIsValid(cur_pos, action):
				v_a.append(action)
		return v_a

	def _getValidActionDict(self):
		v_a = dict()
		for action in self.action:
			if self.checkActionIsValid(self.cur_pos, action):
				v_a.update({action : 1})
			else:
				v_a.update({action : 0})
		return v_a		

	def _getValidActionList(self):
		v_a = []
		for action in self.action:
			if self.checkActionIsValid(self.cur_pos, action):
				v_a.append(action)
		return v_a

	def step(self, action):
		if self.checkActionIsValid(self.cur_pos, action) == True:
			if action == 0:
				self.cur_pos[0] -= 1
			elif action == 1:
				self.cur_pos[0] += 1
			elif action == 2:
				self.cur_pos[1] -= 1
			elif action == 3:
				self.cur_pos[1] += 1
			else:
				raise LookupError('action is invalid, action = ', action, 'cur_pos = ', self.cur_pos)
			if self.cur_pos == self.end_pos:
				self.done = True
			self.reward = self.getRewardAndRemoveCoin(self.cur_pos)
			self.total_reward += self.reward
			observation = np.array((self.coins_permutation_index(), \
									self.pos2state(self.cur_pos)))
			return observation, self.reward, self.done, {}
		else:
			raise LookupError('step function error, action is invalid, action = ', action)

	def reset(self):
		self._init_r_matrix()
		self.done = False
		self.reward = 0
		self.total_reward = 0
		self.cur_pos = self.start_pos[:]
		# print("on reset(): self.cur_pos = ", self.cur_pos, "self.start_pos = ", self.start_pos)
		self.set_agentTrans()
		for state, value in self._coin_geoms.items():
			value[0].set_translation(value[1][0], value[1][1])

	'''
	Here, we map each permuation of coins to a sole index,
	and use the index to indentify different q-table.
	In simpler words, each permutation corresponds to a different q-table.
	Therefore, `coin_idx` is intended as a feature to observe.
	'''
	def coins_permutation_index(self):
		idx = 0
		i = 0
		for key in self.coins:
			res = 1 if self.r_matrix[key] > 0 else 0
			idx |= (res << i)
			i += 1
		return idx

# -------------------------------- output 2 terminal --------------------------

	def action2arrow(self, action, isArrow):
		if isArrow == 1:
			if action == 0:
				return " <--"
			if action == 1:
				return " -->"
			if action == 2:
				return "  \u2191"
			if action == 3:
				return "  \u2193"
		else:
			return ""
	def output_map(self, isArrow=0):
		print("--------------------------------------- Board -------------------------------------------")
		print("|\t\t\t\t\t\t\t\t\t\t\t|")
		for i in range(self.map_width):
			print("|", end = "\t")
			for j in range(self.map_height):
				r = self.r_matrix[self.pos2state([j, i])]
				state = self.pos2state([j, i])
				end_str = "\t"
				for x in self.path:
					if x[0] == state:
						end_str = self.action2arrow(x[1], isArrow) + end_str
						break
				if  r > 0:
					if [j, i] != self.end_pos:
						cprint(r, 'cyan', end = end_str)
					else:
						cprint(state, 'yellow', end = "\t")
				else:
					print(state, end = end_str)
			print("|")
			print("|", end = "")
			print("\t\t\t\t\t\t\t\t\t\t\t|")
		print("------------------------------------------------------------------------------------------")

# --------------------------------- render part --------------------------------
	def render(self, mode='human', close=False):
		# print("render start --------------------")
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		screen_width = 300
		screen_height = 300
		self.cell_width = screen_width / self.map_width
		self.cell_height = screen_height / self.map_height
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			self.draw_map(rendering, screen_width, screen_height, self.cell_width, self.cell_height)
			self.draw_coins(rendering, self.cell_width, self.cell_height)
			self.draw_startingEndPoint(rendering, self.cell_width, self.cell_height)
			self.draw_agent(rendering, self.cell_width, self.cell_height)
		
		self.set_agentTrans()		
		return self.viewer.render(return_rgb_array=mode == 'rgb_array')

	def draw_map(self, rendering, screen_width, screen_height, cell_width, cell_height):
		for i in range(self.map_width + 1):
			line = rendering.Line((0, i*cell_height), (screen_width, i*cell_height))
			line.set_color(0,0,0)
			self.viewer.add_geom(line)
		for i in range(self.map_height + 1):
			line = rendering.Line((i*cell_width, 0), (i*cell_width, screen_height))
			line.set_color(0,0,0)
			self.viewer.add_geom(line)

	def draw_coins(self, rendering, cell_width, cell_height):
		for state in self.coins:
			pivot = self.state2screen(state, cell_width, cell_height)
			coin = rendering.make_circle(15)
			cointrans = rendering.Transform(translation=pivot)
			coin.add_attr(cointrans)
			coin.set_color(1, 0.9, 0)
			self.viewer.add_geom(coin)
			self._coin_geoms[state] = [cointrans, pivot]

	def draw_startingEndPoint(self, rendering, cell_width, cell_height):
		start_pivot = ((self.start_pos[0]+0.5) * cell_width, ((self.map_height-self.start_pos[1])-0.5) * cell_height)
		start_points = self.get_vertex(start_pivot, cell_width, cell_height)
		start_polygon = rendering.make_polygon(start_points)
		start_polygon.set_color(0.8, 0.8, 1)
		self.viewer.add_geom(start_polygon)

		end_pivot = ((self.end_pos[0]+0.5) * cell_width, ((self.map_height-self.end_pos[1])-0.5) * cell_height)
		end_points = self.get_vertex(end_pivot, cell_width, cell_height)
		end_polygon = rendering.make_polygon(end_points)
		end_polygon.set_color(0.8, 0.6, 1)
		self.viewer.add_geom(end_polygon)

	def get_vertex(self, pivot, cell_width, cell_height):
		offset = 1
		v = []
		v.append((pivot[0]-cell_width/2+offset, pivot[1]-cell_height/2+offset))
		v.append((pivot[0]-cell_width/2+offset, pivot[1]+cell_height/2-offset))
		v.append((pivot[0]+cell_width/2-offset, pivot[1]+cell_height/2-offset))
		v.append((pivot[0]+cell_width/2-offset, pivot[1]-cell_height/2+offset))
		return v

	def draw_agent(self, rendering, cell_width, cell_height):
		# 10 and 8.66 are calculated by the triangle properties
		if cell_width < 8.66 or cell_height < 10:
			raise LookupError('cell is too small, can\'t draw traingle(agent)')
		pivot = ((self.cur_pos[0]+0.5) * cell_width, ((self.map_height-self.cur_pos[1])-0.5) * cell_height)
		top_v = (pivot[0], pivot[1]+10)
		bottomLeft_v = (pivot[0]-8.66, pivot[1]-5)
		bottomRight_v = (pivot[0]+8.66, pivot[1]-5)
		v = [top_v, bottomLeft_v, bottomRight_v]
		triangle = rendering.make_polygon(v)
		triangle.set_color(0.9, 0.1, 0.2)
		self.viewer.add_geom(triangle)
		self._agent_geom = triangle

	def set_agentTrans(self):
		# change the three vertexes depends on the pivot screen position
		if self.viewer is not None:
			pivot = ((self.cur_pos[0]+0.5) * self.cell_width, ((self.map_height-self.cur_pos[1])-0.5) * self.cell_height)
			top_v = (pivot[0], pivot[1]+10)
			bottomLeft_v = (pivot[0]-8.66, pivot[1]-5)
			bottomRight_v = (pivot[0]+8.66, pivot[1]-5)
			self._agent_geom.v = [top_v, bottomLeft_v, bottomRight_v]

	def remove_coin(self):
		state = self.pos2state(self.cur_pos)
		# self.viewer.geoms.remove(self._coin_geoms[state])
		self._coin_geoms[state][0].set_translation(-100, -100)

	# return the pivot screen position as the center of the circle
	def state2screen(self, state, cell_width, cell_height):
		pos = self.state2pos(state)
		return ((pos[0]+0.5) * cell_width, ((10-pos[1])-0.5) * cell_height)

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
import gym
import numpy as np
from gym import error, spaces, utils, wrappers
from gym.utils import seeding
from termcolor import colored, cprint


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
		self.action_space = spaces.Discrete(4)
		self.cur_pos = [0, 0]
		self.state = np.arange(self.state_size)
		self.done = 0
		self.reward = 0
		self.r_matrix = np.zeros(100)
		self.reward_state = [18, 62, 66]
		self.reward_value = [4, 6, 10]
		self.start_pos = [0, 0]
		self.end_pos = [5, 5]
		self.path = [(0, 1), (1,1), (2,1), (3,1), (4,3), (14,3), (24,3), (34,1), (35,3), (45,3)]
		self.init_r_matrix()
		self.viewer = None
		self.x=[140,220,300,380,460,140,300,460]
		self.y=[250,250,250,250,250,150,150,150]

	def init_r_matrix(self):
		idx = 0
		for i in range(self.state_size):
			if idx < len(self.reward_state) and i == self.reward_state[idx]:
				self.r_matrix[i] = self.reward_value[idx]
				idx += 1
			elif i == self.pos2state(self.end_pos):
				self.r_matrix[i] = 10
			# print("r_matrix[{0}] = {1}".format(i, i))

	def pos2state(self, pos):
		return pos[1] * self.map_width + pos[0]

	def checkActionIsValid(pos, action):
		if pos[0] == 0 and action == 0:
			return 0
		if pos[0] == self.map_width - 1 and action == 1:
			return 0
		if pos[1] == 0 and action == 2:
			return 0
		if pos[0] == self.map_height - 1 and action == 3:
			return 0
		return 1 

	def getAndRemoveReward(self, pos):
		state = self.pos2state(pos)
		r = self.r_matrix[state]
		if r > 0:
			self.r_matrix[state] = 0

	def step(self, pos, action):
		next_pos = pos
		if(checkActionIsValid(pos, action)):
			if action == 0:
				next_pos[0] -= 1
			elif action == 1:
				next_pos[0] += 1
			elif action == 2:
				next_pos[1] -= 1
			elif action == 3:
				next_pos[1] += 1
			else:
				raise LookupError('action is invalid')
			self.cur_pos = next_pos
			if next_pos == self.end_pos:
				self.done = 1
			self.reward = self.getReward(next_pos)
			return [self.cur_pos, self.reward, self.done]
		else:
			raise LookupError('step function error, action is invalid')

	def reset(self):
		self.init_r_matrix()
		self.done = 0
		self.reward = 0

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

	def render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		screen_width = 600
		screen_height = 400

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			#创建网格世界
			self.line1 = rendering.Line((100,300),(500,300))
			self.line2 = rendering.Line((100, 200), (500, 200))
			self.line3 = rendering.Line((100, 300), (100, 100))
			self.line4 = rendering.Line((180, 300), (180, 100))
			self.line5 = rendering.Line((260, 300), (260, 100))
			self.line6 = rendering.Line((340, 300), (340, 100))
			self.line7 = rendering.Line((420, 300), (420, 100))
			self.line8 = rendering.Line((500, 300), (500, 100))
			self.line9 = rendering.Line((100, 100), (180, 100))
			self.line10 = rendering.Line((260, 100), (340, 100))
			self.line11 = rendering.Line((420, 100), (500, 100))
			#创建第一个骷髅
			self.kulo1 = rendering.make_circle(40)
			self.circletrans = rendering.Transform(translation=(140,150))
			self.kulo1.add_attr(self.circletrans)
			self.kulo1.set_color(0,0,0)
			#创建第二个骷髅
			self.kulo2 = rendering.make_circle(40)
			self.circletrans = rendering.Transform(translation=(460, 150))
			self.kulo2.add_attr(self.circletrans)
			self.kulo2.set_color(0, 0, 0)
			#创建金条
			self.gold = rendering.make_circle(40)
			self.circletrans = rendering.Transform(translation=(300, 150))
			self.gold.add_attr(self.circletrans)
			self.gold.set_color(1, 0.9, 0)
			#创建机器人
			self.robot= rendering.make_circle(30)
			self.robotrans = rendering.Transform()
			self.robot.add_attr(self.robotrans)
			self.robot.set_color(0.8, 0.6, 0.4)

			self.line1.set_color(0, 0, 0)
			self.line2.set_color(0, 0, 0)
			self.line3.set_color(0, 0, 0)
			self.line4.set_color(0, 0, 0)
			self.line5.set_color(0, 0, 0)
			self.line6.set_color(0, 0, 0)
			self.line7.set_color(0, 0, 0)
			self.line8.set_color(0, 0, 0)
			self.line9.set_color(0, 0, 0)
			self.line10.set_color(0, 0, 0)
			self.line11.set_color(0, 0, 0)

			self.viewer.add_geom(self.line1)
			self.viewer.add_geom(self.line2)
			self.viewer.add_geom(self.line3)
			self.viewer.add_geom(self.line4)
			self.viewer.add_geom(self.line5)
			self.viewer.add_geom(self.line6)
			self.viewer.add_geom(self.line7)
			self.viewer.add_geom(self.line8)
			self.viewer.add_geom(self.line9)
			self.viewer.add_geom(self.line10)
			self.viewer.add_geom(self.line11)
			self.viewer.add_geom(self.kulo1)
			self.viewer.add_geom(self.kulo2)
			self.viewer.add_geom(self.gold)
			self.viewer.add_geom(self.robot)

		if self.state is None: return None
		#self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
		# self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])



		return self.viewer.render(return_rgb_array=mode == 'rgb_array')
		
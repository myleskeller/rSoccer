import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseFIRAEnv
from rsoccer_gym.Utils import KDTree

from rsoccer_gym.vss.env_motion_tuning.univectorPosture import UnivectorPosture
from rsoccer_gym.vss.env_motion_tuning.goToBallState import goToBallState

def distance(pointA, pointB):

    x1 = pointA[0]
    y1 = pointA[1]

    x2 = pointB[0]
    y2 = pointB[1]

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return distance


class VSSMotionTuningEnv(VSSBaseFIRAEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation:
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=1, n_robots_yellow=1,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        obsSpaceThresholds = np.array([0.75, 0.65, 0.75, 0.65, 2, 2, math.pi * 3, 10], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds,
                                                dtype=np.float32)
        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = None
        self.v_wheel_deadzone = 0.05
        print(self.observation_space.shape)

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )
        
        self.goToballState = None
        self.distAnt = 1000
        self.angleAnt = 0
        self.timestampAnt = 0
        self.target = None

        self.v_max = None
        self.v_max_min = 0.7
        self.v_max_max = 1.1

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None
        self.randomize()
        #print(self.rand_params)
        for ou in self.ou_actions:
            ou.reset()
        for i in range(10):
          data = super().reset()
        self.path = self._run_planning(self.frame, 0, False)
        self.goToballState=goToBallState()
        return self._frame_to_observations()
    
    def randomize(self):
        v_max = np.random.uniform(self.v_max_min, self.v_max_max)
        self.rand_params = []
        self.rand_params.append(v_max)


    def step(self, action):
        #observation, reward, done, _ = super().step(action)
        #return observation, reward, done, self.reward_shaping_total
        return super().step(action)

    def _frame_to_observations(self):

        observation = []
        if(len(self.path)<2):
          observation = self.goToballState.getObservation(self.frame, [(self.frame.ball.x, self.frame.ball.y), (self.frame.ball.x, self.frame.ball.y)])
        else:
          observation = self.goToballState.getObservation(self.frame, self.path)
    

        #observation.append(self.norm_pos(self.goToballState.ball_x))
        #observation.append(self.norm_pos(self.goToballState.ball_y))
        #observation.append(self.norm_v(self.goToballState.robot_vx))
        #observation.append(self.norm_v(self.goToballState.robot_vy))
        #observation.append(self.norm_w(self.goToballState.robot_w))
        #observation.append(self.goToballState.distance)


        #observation.append(self.norm_pos(self.path[0][0]))
        #observation.append(self.norm_pos(self.path[0][1]))
        ##observation.append(self.norm_v(self.frame.ball.v_x))
        ##observation.append(self.norm_v(self.frame.ball.v_y))
        #
        #for i in range(self.n_robots_blue):
        #    observation.append(self.norm_pos(self.frame.robots_blue[i].x))
        #    observation.append(self.norm_pos(self.frame.robots_blue[i].y))
        #    observation.append(
        #        np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
        #    )
        #    observation.append(
        #        np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
        #    )
        #    observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
        #    observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
        #    observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))
        #
        ##for i in range(self.n_robots_yellow):
        #    observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
        #    observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
        #    observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
        #    observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
        #    observation.append(
        #        self.norm_w(self.frame.robots_yellow[i].v_theta)
        #    )

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}
        #print(actions)
        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0,
                              v_wheel1=v_wheel1))
        
        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = (0,0)
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        for i in range(self.n_robots_yellow):
            actions = (0,0)
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i, v_wheel0=v_wheel0,
                                  v_wheel1=v_wheel1))
        #print(commands)
        return commands

    def _calculate_reward_and_done(self):
      reward = 0
      rewardContact = 0
      rewardDistance = 0
      done = False

      pathDistance = distance((self.frame.robots_blue[0].x, self.frame.robots_blue[0].y), self.path[0])

      if(pathDistance<0.1):
        #print("Chegou")
        rewardContact += 1
        if(len(self.path)>0):
          self.path.pop(0)
        #print(self.path)
       
      rewardDistance -= 1.0/250.0
      #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance)**2 + self.goToBallState.angle_relative**2) / 2) - 2


      if  self.steps >= 250:
        #finished the episode
        done = True
      else:
        # the ball in the field limits
        if (len(self.path)==0):
          #print("OI")
          rewardContact += 1
          done = True
        #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2 

      reward = rewardContact + rewardDistance
      #print(self.path[len(self.path)-1])
      return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return np.deg2rad(random.uniform(0, 360))

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            
            pos = (1.0, 1.0)
            #print(places.get_nearest(pos)[1] < min_dist)
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            #print(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        #print("max_v", self.max_v)
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )
        #print("left ", left_wheel_speed, self.field.rbt_wheel_radius)
        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed , right_wheel_speed
      
    def _run_planning(self, frame, index, yellow):
        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1

        ball = frame.ball
        #print("Ball ", ball)
        if(yellow):
          robot = frame.robots_yellow[index]
        else :
          robot=frame.robots_blue[index]

        if yellow:
          #angle_rob = np.deg2rad(robot.theta)
          robot_pos = ((lenght + robot.x)*100, (width + robot.y) * 100)
          ball_pos = ((lenght + ball.x) * 100, (width + ball.y) * 100)
          ball_speed = (ball.v_x * 100, ball.v_y * 100)
          allies = []
          for i in range(len(frame.robots_yellow)):
              robot = frame.robots_yellow[i]
              allies.append(((lenght + robot.x) * 100, (width + robot.y) * 100))
          enemies = []
          for i in range(len(frame.robots_blue)):
              robot = frame.robots_blue[i]
              enemies.append(((lenght + robot.x) * 100, (width + robot.y) * 100))
        else:
          #print("caaa", robot.x, robot.y)

          #angle_rob = robot.orientation + math.pi
          #if angle_rob > math.pi:
          #    angle_rob -= 2*math.pi
          #elif angle_rob < -math.pi:
          #    angle_rob += 2*math.pi
          robot_pos = ((lenght - robot.x) * 100,(width - robot.y) * 100)
          ball_pos = ((lenght -ball.x) * 100, (width - ball.y) * 100)
          ball_speed = (-ball.v_x * 100, -ball.v_y * 100)
          allies = []
          for i in range(len(frame.robots_blue)):
              robot = frame.robots_blue[i]
              allies.append(((lenght - robot.x) * 100,(width - robot.y) * 100))
          enemies = []
          for i in range(len(frame.robots_yellow)):
              robot = frame.robots_yellow[i]
              enemies.append(((lenght - robot.x) * 100,(width - robot.y) * 100))

        #print(angle_rob)
        #print(robot_pos)
        univector = UnivectorPosture()
        path = univector.update(ball_pos,robot_pos, ball_pos,allies,enemies,index)
        #print(path)
        for i in range (len(path)):
          if yellow:
            path[i] = (path[i][0]/100 - lenght, path[i][1]/100 - width) 
          else:
            path[i] = (lenght - path[i][0]/100, width - path[i][1]/100)
        #print("NEW", path)
        #circle = Circle(0.5)
        #path = circle.discretization(40)
        return path



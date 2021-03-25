import gym
import math
import numpy as np
import time
import random

#from gym_tuning.fira_env import FiraEnv
#from libs.fira_client import FiraClient
from rc_gym.vss.vss_gym_base import VSSBaseEnv
from rc_gym.Entities import Robot, Ball, Frame
from rc_gym.vss.env_motion_control.goToBallState import goToBallState
import rc_gym.vss.env_motion_control.Utils as Utils
#from rc_gym.Utils import *

class goToBallEnv(VSSBaseEnv):
  """
  Using cartpole env description as base example for our documentation
  Description:
      # TODO
  Source:
      # TODO

  Observation:
      Type: Box(14)
      Num     Observation                                       Min                     Max
      0       Ball X   (m)                                   -7000                   7000
      1       Ball Y   (m)                                   -6000                   6000
      2       Ball X   (m)                                   -7000                   7000
      3       Ball Y   (m)                                   -6000                   6000
      4       Blue id 0 Vx  (m/s)                            -10000                  10000
      5       Blue id 0 Vy  (m/s)                            -10000                  10000
      6       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
      7       Dist Blue id0 - ball (m)                       -10000                  10000
      
      

  Actions:
      Type: Box(2)
      Num     Action                        Min                      Max
      0       Vl                           -100                      100
      1       Vr                           -100                      100
  Reward:
      Reward is 1 for success, -1 to fails. 0 otherwise.

  Starting State:
      All observations are assigned a uniform random value in [-0.05..0.05]
      # TODO

  Episode Termination:
    # TODO
  """
  def __init__(self):
    super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                     time_step=0.032)
    ## Action Space
    actSpaceThresholds = np.array([1, 1], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-1, high=1,
                                        shape=(2, ), dtype=np.float32)


    # Observation Space thresholds
    obsSpaceThresholds = np.array([0.75, 0.65, 0.75, 0.65, 2, 2, math.pi * 3, 10], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.goToballState = None
    self.distAnt = 1000
    self.angleAnt = 0
    self.timestampAnt = 0
    self.target = None
    print('Environment initialized')


  
  def _get_commands(self, actions):
    commands = []
    v_wheel1 = actions[0]
    v_wheel2 = actions[1]
    #self.energy_penalty = -(abs(v_wheel1 * 100) + abs(v_wheel2 * 100))
    commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                          v_wheel2=v_wheel2))
    return commands

  def _frame_to_observations(self):
    observation = []

    self.goToBallState = goToBallState()

    observation = self.goToBallState.getObservation(self.frame, self.path)

    return np.array(observation)

  def _get_initial_positions_frame(self):
    posFrame = Frame()
    #To CHANGE: 
    # ball penalty position
    #ball = Ball(x=random.uniform(-4, 0), y=random.uniform(-4, 4), v_x=0, v_y=0)
    posFrame.ball.x =random.uniform(-0.5,0) 
    posFrame.ball.y=random.uniform(-0.4, 0.4)
    


    #ball = Ball(x=0.75, y=0)
    
    # Goalkeeper penalty position
    #goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)
    #goalKeeper = Robot(id=0, x=6, y=1, theta=0, yellow = True)

    # Kicker penalty position
    #attacker = Robot(id=0, x=random.uniform(-3.5, 0), y=random.uniform(-4, 4), theta=180, yellow = False)
    posFrame.robots_blue[0] = Robot(id=0, x=random.uniform(-0.7,0), y=random.uniform(-0.2,0.2), theta=180, yellow = False)
    posFrame.robots_blue[1] = Robot(x=-1.0, y=0, theta=0, yellow = False)
    posFrame.robots_blue[2] = Robot(x=-1.0, y=1.0, theta=0, yellow = False)
    posFrame.robots_yellow[0] = Robot(x=1.0, y=-1.0, theta=math.pi, yellow = True)
    posFrame.robots_yellow[1] = Robot(x=1.0, y=0, theta=math.pi, yellow = True)
    posFrame.robots_yellow[2] = Robot(x=1.0, y=1.0, theta=math.pi, yellow = True)
    self.goToBallState = goToBallState()
    self.path = self.goToBallState.generatePath(posFrame)

    return posFrame
    
  def _calculate_reward_and_done(self):
    reward = 0
    rewardContact = 0
    rewardDistance = 0
    rewardAngle =0
    done = False

    if(self.goToBallState.distance<0.1):
      rewardContact += 100
      self.path.pop(0)
    #print(self.state.timestamp)

    
    #dt = self.data.step - self.timestampAnt
    #
    #if(dt==0):
    #  dt =1

    rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance)**2 + self.goToBallState.angle_relative**2) / 2) - 2
    #rewardDistance -= (0.1*(self.goToBallState.distance - self.distAnt))/dt
    #rewardAngle -=(0.02*(self.goToBallState.angle_relative - self.angleAnt))/dt

    self.distAnt = self.goToBallState.distance
    self.timestampAnt = self.step
    self.angleAnt = self.goToBallState.angle_relative
    #print(self.goToBallState.distance)
    if self.frame.ball.x < -0.75 or self.frame.ball.y > 0.65 or self.frame.ball.y < -0.65 or  self.frame.ball.x > 0.75:
      # the ball out the field limits
      done = True
      rewardContact += 0
      
    elif  self.steps > 250:
      #finished the episode
      done = True
      #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2
    else:
      # the ball in the field limits
      if (len(self.path)==0):
        #print("OI")
        rewardContact += 500
        done = True
      #rewardDistance += (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((self.goToBallState.distance*0.001)**2 + self.goToBallState.angle_relative**2) / 2) - 2 

    reward = rewardContact + rewardDistance
    #print(self.path[len(self.path)-1])
    return reward, done
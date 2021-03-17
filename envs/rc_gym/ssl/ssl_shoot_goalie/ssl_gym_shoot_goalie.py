import math
import random
from rc_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv
from rc_gym.ssl.ssl_shoot_goalie.shootGoalieState import *
from rc_gym.ssl.ssl_shoot_goalie.geometry import mod


class SSLShootGoalieEnv(SSLBaseEnv):
    """
   Description:
     SSL Robot scoaring goal with a mobile GoalKeeper
   Observation:
     Type: Box(16)
     Num     Observation                                       Min                     Max
     0       Ball X   (m)                                  -field_lenth/2           field_lenth/2
     1       Ball Y   (m)                                  -field_width/2           field_width/2
     2       Ball Vx  (m/s)                                  -6.5 m/s                 6.5 m/s
     3       Ball Vy  (m/s)                                  -6.5 m/s                 6.5 m/s
     4       Blue id 0 Robot Vw       (rad/s)                -10 rad/s                10 rad/s
     5       Dist Blue id0 - goal center (m)                   0            norm(field_lenth, field_width)
     6       Angle between blue id 0 and goal left (cos)       -1                         1
     7       Angle between blue id 0 and goal left (sin)       -1                         1
     8       Angle between blue id 0 and goal right (cos)      -1                         1
     9       Angle between blue id 0 and goal right (sin)      -1                         1
     10      Angle between blue id 0 and goalie center(cos)    -1                         1
     11      Angle between blue id 0 and goalie center(sin)    -1                         1
     12      Angle between blue id 0 and goalie left (cos)     -1                         1
     13      Angle between blue id 0 and goalie left (sin)     -1                         1
     14      Angle between blue id 0 and goalie right (cos)    -1                         1
     15      Angle between blue id 0 and goalie right (sin)    -1                         1
   Actions:
     Type: Box(2)
     Num     Action                        Min                     Max
     0       Blue id 0 Vw (rad/s)       -10 rad/s                10 rad/s
     1       Blue Kick Strength (m/s)     -6.5 m/s                6.5 m/s
   Reward:
     Reward is 2 for goal, -0.3 for ball outside of field, -0.01 otherwise.
   Episode Termination:
     # Goal, or ball outside of field.
   """

    def __init__(self, field_type=1):
        super().__init__(field_type=field_type, n_robots_blue=1,
                         n_robots_yellow=1, time_step=0.032)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)

        n_obs = 16
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        self.shootGoalieState = shootGoalieState(field_params=self.field_params)

        print('Environment initialized')

    def _frame_to_observations(self):

        ball_x, ball_y = self.shootGoalieState.getBallLocalCoordinates(self.frame)
        ball_vx, ball_vy = self.shootGoalieState.getBallLocalSpeed(self.frame)
        
        distance = self.shootGoalieState.getDistance(self.frame)
        robot_w = self.frame.robots_blue[0].v_theta
        theta_l_sen, theta_l_cos = self.shootGoalieState.getLeftPoleAngle(self.frame)
        theta_r_sen, theta_r_cos = self.shootGoalieState.getRightPoleAngle(self.frame)
        theta_goalie_c_sen, theta_goalie_c_cos = self.shootGoalieState.getGoalieCenterAngle(self.frame)
        theta_goalie_l_sen, theta_l_cos = self.shootGoalieState.getGoalieLeftAngle(self.frame)
        theta_goalie_r_sen, theta_r_cos = self.shootGoalieState.getGoalieRightAngle(self.frame)
        
        observation = []

        observation.append(self.norm_dist(distance))
        observation.append(self.norm_pos(ball_x)) 
        observation.append(self.norm_pos(ball_y)) 
        observation.append(self.norm_v(ball_vx)) 
        observation.append(self.norm_v(ball_vy)) 
        observation.append(self.norm_w(robot_w))
        observation.append(theta_l_sen)
        observation.append(theta_l_cos)
        observation.append(theta_r_sen) 
        observation.append(theta_r_cos)
        observation.append(theta_goalie_c_sen)
        observation.append(theta_goalie_c_cos)
        observation.append(theta_goalie_l_sen)
        observation.append(theta_l_cos)
        observation.append(theta_goalie_r_sen)
        observation.append(theta_r_cos)
        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        commands.append(
            Robot(yellow=False, id=0, v_theta=actions[0], kick_v_x= 1 if actions[1] > 0.2 else 0, dribbler=True))
        
        
        # Moving GOALIE
        goal_width = self.field_params['goal_width']
        vy = (self.frame.ball.y - self.frame.robots_yellow[0].y)
        if abs(vy) > 0.4:
            vy = 0.4*(vy/abs(vy))
        if self.frame.robots_yellow[0].y > goal_width/2-0.1 and vy > 0:
            vy = 0
        if self.frame.robots_yellow[0].y < -goal_width/2+0.1 and vy < 0:
            vy = 0

        cmdGoalie = self._getCorrectGKCommand(vy)
        print (cmdGoalie)
        commands.append(cmdGoalie)

        return commands

    def _calculate_reward_and_done(self):
        return self._penalizeRewardFunction()

    def _sparseReward(self):
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        goal_width = self.field_params['goal_width']

        reward = 0
        done = False

        if self.frame.ball.x < -field_half_length:
            # the ball out the field limits
            done = True
            if self.frame.ball.y < goal_width/2 and self.frame.ball.y > -goal_width/2:
                # ball entered the goal
                reward = 1
            else:
                # the ball went out the bottom line
                reward = -1
        elif self.frame.ball.x < -(field_half_length-1) and self.frame.ball.v_x > -0.01:
            # goalkeeper caught the ball
            done = True
            reward = -1
        elif mod(self.frame.ball.v_x, self.frame.ball.v_y) < 0.01 and self.steps > 15:
            # 1 cm/s
            done = True
            reward = -1

        return reward, done

    def _penalizeRewardFunction(self):
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        goal_width = self.field_params['goal_width']

        reward = -0.01
        done = False
        if self.frame.ball.x < -field_half_length:
            # the ball out the field limits
            done = True
            if self.frame.ball.y < goal_width/2 and self.frame.ball.y > -goal_width/2:
                # ball entered the goal
                reward = 2
        elif self.frame.ball.x < -(field_half_length-1) and self.frame.ball.v_x > -0.01:
            # goalkeeper caught the ball
            done = True
            reward = -0.3

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the inicial frame'''

        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        goal_width = self.field_params['goal_width']

        pos_frame: Frame = Frame()

        attacker_x = -field_half_length+2
        attacker_y = random.uniform(-field_half_width/2, field_half_width/2)
        robot_theta = random.uniform(-180, 180)

        ball_x = attacker_x - 0.09*math.cos(math.radians(robot_theta+180))
        ball_y = attacker_y - 0.09*math.sin(math.radians(robot_theta+180))

        pos_frame.ball.x = ball_x
        pos_frame.ball.y = ball_y
        pos_frame.ball.v_x = 0
        pos_frame.ball.v_y = 0

        # Goalkeeper position
        goal_keeper_y = random.uniform(-goal_width/2, goal_width/2)
        goal_keeper = Robot(id=0, x=-field_half_length,
                            y=goal_keeper_y, theta=0, yellow=True)
        # Kicker position
        attacker = Robot(id=0, x=attacker_x, y=attacker_y,
                         theta=robot_theta, yellow=False, kick_v_x=0, dribbler=True)

        # For fixed positions!
        # ball = Ball(x=-4.1, y=0, v_x=0, vy=0)
        # # Goalkeeper position
        # goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)
        # # Kicker position
        # attacker = Robot(id=0, x=-4, y=0, theta=180, yellow = False)

        pos_frame.robots_blue[0] = attacker
        pos_frame.robots_yellow[0] = goal_keeper

        return pos_frame

    def _getCorrectGKCommand(self, v_y):
        '''Control goalkeeper v_theta and vx to keep him at goal line'''
        field_half_length = self.field_params['field_length'] / 2

        cmdGoalKeeper = Robot(yellow=True, id=0, v_y=v_y)

        # Proportional Parameters for Vx and Vw
        KpVx = 0.0006
        KpVw = 0.005
        # Error between goal line and goalkeeper
        errX = -field_half_length - self.frame.robots_yellow[0].x
        # If the error is greater than 20mm, correct the goalkeeper
        if abs(errX) > 20:
            cmdGoalKeeper.v_x = KpVx * errX
        else:
            cmdGoalKeeper.v_x = 0.0
        # Error between the desired angle and goalkeeper angle
        errW = 0.0 - abs(self.frame.robots_yellow[0].theta)
        # If the error is greater than 0.1 rad (5,73 deg), correct the goalkeeper
        if abs(errW) > 6:
            cmdGoalKeeper.v_theta = KpVw * errW
        else:
            cmdGoalKeeper.v_theta = 0.0

        return cmdGoalKeeper

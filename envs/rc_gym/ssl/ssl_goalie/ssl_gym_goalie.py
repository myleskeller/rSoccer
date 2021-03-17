import math
import random
from rc_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot, Ball
from rc_gym.ssl.ssl_gym_base import SSLBaseEnv
from rc_gym.ssl.ssl_goalie.goalieState import *
from rc_gym.ssl.ssl_goalie.geometry import mod


class SSLGoalieEnv(SSLBaseEnv):
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
     4       Blue id 0 Robot X  (m)                       -field_lenth/2           field_lenth/2
     5       Blue id 0 Robot Y  (m)                       -field_width/2           field_width/2
     6       Dist Blue id0 - goal center (m)                   0            norm(field_lenth, field_width)
     7       Angle between blue id 0 and goal left (cos)       -1                         1
     8       Angle between blue id 0 and goal left (sin)       -1                         1
     9       Angle between blue id 0 and goal right (cos)      -1                         1
     10      Angle between blue id 0 and goal right (sin)      -1                         1
     11      Angle between blue id 0 and ball (cos)            -1                         1
     12      Angle between blue id 0 and ball (sin)            -1                         1
     13      Angle between blue id 0 and yellow id 0 (cos)     -1                         1
     14      Angle between blue id 0 and yellow id 0 (sin)     -1                         1
     15      Angle of yellow id 0 (sin)                        -1                         1
     16      Angle of yellow id 0 (cos)                        -1                         1
   Actions:
     Type: Box(2)
     Num     Action                        Min                     Max
     0       Blue id 0 Vw (rad/s)       -10 rad/s                10 rad/s
     1       Blue id 0 Vx (m/s)         -6.5 m/s                6.5 m/s
     2       Blue id 0 Vy (m/s)         -6.5 m/s                6.5 m/s
   Reward:
     Reward is 2 for goal, -0.3 for ball outside of field, -0.01 otherwise.
   Episode Termination:
     # Goal, or ball outside of field.
   """

    def __init__(self, field_type=1):
        super().__init__(field_type=field_type, n_robots_blue=1,
                         n_robots_yellow=1, time_step=0.032)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(3, ), dtype=np.float32)

        n_obs = 17
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs, ),
                                                dtype=np.float32)

        self.goalieState = goalieState(field_params=self.field_params)

        print('Environment initialized')

    def _frame_to_observations(self):

        ball_x, ball_y = self.goalieState.getBallLocalCoordinates(self.frame)
        ball_vx, ball_vy = self.goalieState.getBallLocalSpeed(self.frame)
        
        distance = self.goalieState.getDistance(self.frame)
        robot_x = self.frame.robots_blue[0].x
        robot_y = self.frame.robots_blue[0].y
        theta_l_sen, theta_l_cos = self.goalieState.getLeftPoleAngle(self.frame)
        theta_r_sen, theta_r_cos = self.goalieState.getRightPoleAngle(self.frame)
        theta_ball_sen, theta_ball_cos = self.goalieState.getBallAngle(self.frame)
        theta_attacker_sen, theta_attacker_cos = self.goalieState.getAttackerAngle(self.frame)
        
        observation = []

        observation.append(self.norm_pos(ball_x)) 
        observation.append(self.norm_pos(ball_y)) 
        observation.append(self.norm_v(ball_vx)) 
        observation.append(self.norm_v(ball_vy)) 
        observation.append(self.norm_pos(robot_x))
        observation.append(self.norm_pos(robot_y))
        observation.append(self.norm_dist(distance))
        observation.append(theta_l_sen)
        observation.append(theta_l_cos)
        observation.append(theta_r_sen) 
        observation.append(theta_r_cos)
        observation.append(theta_ball_sen)
        observation.append(theta_ball_cos)
        observation.append(theta_attacker_sen)
        observation.append(theta_attacker_cos)
        observation.append(math.cos(self.frame.robots_yellow[0].theta))
        observation.append(math.sin(self.frame.robots_yellow[0].theta))
        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        commands.append(
            Robot(yellow=True, id=0, v_theta=0, kick_v_x=0.6, dribbler=True))
        
        commands.append(
            Robot(yellow=False, id=0, v_theta=actions[0], v_x = actions[1], v_y= actions[2], kick_v_x=0, dribbler=True))

        return commands

    def _calculate_reward_and_done(self):
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
                reward = -1
            else:
                # the ball went out the bottom line
                reward = 1
        elif self.frame.ball.x < -(field_half_length-1) and self.frame.ball.v_x > -0.01:
            # goalkeeper caught the ball
            done = True
            reward = 1
        elif mod(self.frame.ball.v_x, self.frame.ball.v_y) < 0.01 and self.steps > 15:
            # 1 cm/s
            done = True
            reward = 0

        return reward, False

    

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the inicial frame'''

        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2
        goal_width = self.field_params['goal_width']

        pos_frame: Frame = Frame()

        attacker_x = -field_half_length+3
        attacker_y = random.uniform(-field_half_width/2, field_half_width/2)
        kick_y = random.uniform(-goal_width/3, goal_width/3)
        attacker_theta = math.degrees(self.goalieState.getUnifiedAngle(-field_half_length, attacker_x, kick_y, attacker_y, -math.pi))

        ball_x = attacker_x - 0.09*math.cos(math.radians(attacker_theta+180))
        ball_y = attacker_y - 0.09*math.sin(math.radians(attacker_theta+180))

        pos_frame.ball.x = ball_x
        pos_frame.ball.y = ball_y
        pos_frame.ball.v_x = 0
        pos_frame.ball.v_y = 0

        # Kicker position
        attacker = Robot(id=0, x=attacker_x, y=attacker_y,
                         theta=attacker_theta, yellow=True, kick_v_x=0, dribbler=True)

        # Goalkeeper position
        goal_keeper_y = random.uniform(-goal_width/2, goal_width/2)
        goal_keeper = Robot(id=0, x=-field_half_length,
                            y=goal_keeper_y, theta=0, yellow=False)
       
        pos_frame.robots_blue[0] = goal_keeper
        pos_frame.robots_yellow[0] = attacker

        return pos_frame
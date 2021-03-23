from controleDefender import PID
import utils
import math
from goleiro import GoalKeeperDeterministic

from firaInterface.fira_parser import *

gk = GoalKeeperDeterministic()
p = PID()

fira = FiraParser()
fira.start()

while(1):
    state = fira.receive()

    ball = state.balls[0]

    width = 1.3/2.0
    lenght = (1.5/2.0) + 0.1

    '''if(self.is_team_yellow):
        pose.x = 
        pose.y = 
        pose.yaw = math.pi - pose.yaw
        v_pose.x *= -100
        v_pose.y *= 100
        v_pose.yaw = v_pose.yaw
    else:
        pose.x = (lenght+pose.x)*100
        pose.y = (width - pose.y)*100
        pose.yaw = -pose.yaw
        v_pose.x *= 100
        v_pose.y *= -100
        v_pose.yaw = v_pose.yaw'''

    robot = state.robots_yellow[0]
    angle_rob = -(robot.pose.yaw-math.pi)

    robot_pos = ((lenght - robot.pose.x)+170, -(width - robot.pose.y))
    ball_pos = ((lenght - ball.pose.x)+170, (width + ball.pose.y))
    ball_speed = (ball.v_pose.x*100,ball.v_pose.y*100)
    #print(robot_pos)
    #print(ball_pos)
    #print(ball_speed)
    #print(angle_rob)

    obj_pos = gk.decideAction(ball_pos,robot_pos,ball_speed)
    print(obj_pos)
    if(obj_pos == None):
        speeds = utils.spin(robot_pos,ball_pos,ball_speed)
        print(speeds)
        fira.send_speeds(speeds[0],speeds[1])

    else:
        speeds = p.run(angle_rob,obj_pos,robot_pos)
        print(speeds[0],speeds[1])
        fira.send_speeds(speeds[0],speeds[1])

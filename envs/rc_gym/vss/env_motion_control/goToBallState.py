from dataclasses import dataclass
from rc_gym.vss.env_motion_control.Utils import *
import rc_gym.vss.env_motion_control.deterministic_vss.univector as univector

TOP_FIELD = 0.75
BOTTOM_FIELD = -0.75
LEFT_FIELD = -0.65
RIGHT_FIELD = 0.65

@dataclass
class goToBallState:
  """Init Frame object."""
  ball_x: float = None
  ball_y: float = None
  target_0_x: float = None
  target_0_y: float = None
  target_1_x: float = None
  target_1_y: float = None
  robot_vx: float = None
  robot_vy: float = None
  robot_w: float = None
  distance: float = None
  wall_top_x: float = None
  wall_top_y: float = None
  wall_bottom_x: float = None
  wall_bottom_y: float = None
  wall_left_x: float = None
  wall_left_y: float = None
  wall_right_x: float = None
  wall_right_y: float = None
  angle_relative: float = None
  

  def getDistance(self, frame, target_pos) -> float:
    #print(float(mod(abs(frame.robots_blue[0].x-frame.ball.x), abs(frame.robots_blue[0].y-frame.ball.y))))
    return float(mod(abs(frame.robots_blue[0].x-target_pos[0]), abs(frame.robots_blue[0].y-target_pos[1])))

  def getTopPosition(self, frame):
    diff_y = TOP_FIELD - frame.robots_blue[0].y
    pos_x = math.sin(frame.robots_blue[0].theta) * diff_y
    pos_y = math.cos(frame.robots_blue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y

  def getBottomPosition(self, frame):
    diff_y = BOTTOM_FIELD - frame.robots_blue[0].y
    pos_x = math.sin(frame.robots_blue[0].theta) * diff_y
    pos_y = math.cos(frame.robots_blue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y

  def getLeftPosition(self, frame):
    diff_y = LEFT_FIELD - frame.robots_blue[0].x
    pos_x = math.cos(frame.robots_blue[0].theta) * diff_y
    pos_y = -math.sin(frame.robots_blue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y 

  def getRightPosition(self, frame):
    diff_y = RIGHT_FIELD - frame.robots_blue[0].x
    pos_x = math.cos(frame.robots_blue[0].theta) * diff_y
    pos_y = -math.sin(frame.robots_blue[0].theta) * diff_y
    #print(pos_x, pos_y)
    return pos_x, pos_y  

  def getRelativeRobotToBallAngle(self, frame, target_pos):
    #dist_left = [abs(frame.ball.x - frame.robots_blue[0].x), abs(frame.ball.y - frame.robots_blue[0].y)]
    #angle_ball = angle(dist_left[0], dist_left[1])
    # print(angle_ball)
    # if frame.robots_blue[0].theta * frame.ball.y >=0:
    #   # the direction of the robot and the position of the ball are both in the same side of the fild (top or bottom)
    #   angle_relative = 
    # print(frame.robots_blue[0].theta)
    #sign = lambda x: (1, -1)[x<0]
    #if frame.ball.x <= frame.robots_blue[0].x:
    #
    #  if sign(frame.ball.y) ^ sign(frame.robots_blue[0].y) >= 0:
    #    #if both values have the same signal
    #    angle_relative = abs(angle_ball - (math.pi - abs(frame.robots_blue[0].theta)))
    #  else:
    #    angle_relative = abs(angle_ball + (math.pi - abs(frame.robots_blue[0].theta)))
    #else:
    #  if sign(frame.ball.y) ^ sign(frame.robots_blue[0].y) >= 0: 
    #    angle_relative = abs(abs(frame.robots_blue[0].theta) - angle_ball)
    #  else:
    #    angle_relative = abs(abs(frame.robots_blue[0].theta) + angle_ball)
    #
    #print(angle_relative)
    #
    robot_ball = [frame.robots_blue[0].x - target_pos[0], frame.robots_blue[0].y - target_pos[1]]
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robots_blue[0].theta))
    #print(angle_to_ball)
    return angle_to_ball

  def getBallLocalCoordinates(self, frame, target):
    robot_ball = [frame.robots_blue[0].x - target[0], frame.robots_blue[0].y - target[1]]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robots_blue[0].theta))
    robot_ball_x = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_y = mod_to_ball* math.sin(angle_to_ball)
    #print(robot_ball_x, robot_ball_y)
    return robot_ball_x, robot_ball_y
  
  def getBallLocalSpeed(self, frame):
    robot_ball = [frame.robots_blue[0].v_x - frame.ball.v_x, frame.robots_blue[0].v_y - frame.ball.v_y]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robots_blue[0].theta))
    robot_ball_vx = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_vy = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_vx, robot_ball_vy


    
  
  def getObservation(self, frame, target_pos):
    frame.robots_blue[0].theta = np.deg2rad(frame.robots_blue[0].theta)
    if frame.robots_blue[0].theta > math.pi:
      frame.robots_blue[0].theta -= 2*math.pi
    elif frame.robots_blue[0].theta < -math.pi:
      frame.robots_blue[0].theta += 2*math.pi
    #frame.robots_blue[0].v_theta = np.deg2rad(frame.robots_blue[0].v_theta)
    #print(frame.robots_blue[0].theta)
   
    self.target_0_x, self.target_0_y = self.getBallLocalCoordinates(frame, target_pos[0])
    if(len(target_pos)>1):
      self.target_1_x, self.target_1_y = self.getBallLocalCoordinates(frame, target_pos[1])
    else:
      self.target_1_x, self.target_1_y = self.getBallLocalCoordinates(frame, target_pos[0])

    #self.ball_x, self.ball_y = target[0], target[1]
    #print(self.ball_x, self.ball_y)
    #self.ball_x, self.ball_y = frame.ball.x, frame.ball.y
    self.ball_vx, self.ball_vy = self.getBallLocalSpeed(frame)
    self.robot_vx = frame.robots_blue[0].v_x
    self.robot_vy = frame.robots_blue[0].v_y
    
    self.distance = self.getDistance(frame, target_pos[0])
    self.robot_w = frame.robots_blue[0].v_theta
    self.wall_top_x, self.wall_top_y = self.getTopPosition(frame)
    self.wall_bottom_x, self.wall_bottom_y = self.getBottomPosition(frame)   
    self.wall_left_x, self.wall_left_y = self.getLeftPosition(frame)
    self.wall_right_x, self.wall_right_y = self.getRightPosition(frame)
    self.angle_relative = self.getRelativeRobotToBallAngle(frame, target_pos[0])
    #print(self.angle_relative)

    
    
    observation = []
    #objective_pos = self.run_planning(frame,0,True)
    #observation.append(self.ball_x) 
    #observation.append(self.ball_y)
    observation.append(self.target_0_x)
    observation.append(self.target_0_y)
    #observation.append(self.target_1_x)
    #observation.append(self.target_1_y)
    observation.append(self.robot_vx) 
    observation.append(self.robot_vy) 
    observation.append(self.robot_w)
    observation.append(self.distance)
    #observation.append(self.wall_top_x)
    #observation.append(self.wall_top_y)
    #observation.append(self.wall_bottom_x) 
    #observation.append(self.wall_bottom_y)
    #observation.append(self.wall_left_x)
    #observation.append(self.wall_left_y)
    #observation.append(self.wall_right_x)
    #observation.append(self.wall_right_y)
    
    return observation

  def generatePath(self, frame):
    objective_pos = self.run_planning(frame,0,True)
    #for i in range(len(objective_pos)):
    #  objective_pos[i] = self.getBallLocalCoordinates(frame)
    return objective_pos


  def run_planning(self, frame, index, yellow):
    width = 1.3/2.0
    lenght = (1.5/2.0) + 0.1

    ball = frame.ball
    #print(frame)
    robot = frame.robots_yellow[index] if yellow else frame.robots_blue[index]

    if yellow:
        angle_rob = np.deg2rad(robot.theta)
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
        angle_rob = np.deg2rad(robot.theta + 180)
        if angle_rob > math.pi:
            angle_rob -= 2*math.pi
        elif angle_rob < -math.pi:
            angle_rob += 2*math.pi

        robot_pos = ((lenght - robot.x) * 100,(width - robot.y) * 100)
        ball_pos = ((lenght -ball.x) * 100, (width - ball.y) * 100)
        ball_speed = (-ball.v_x * 100, -ball.v_y * 100)

        allies = []
        for i in frame.robots_blue:
            robot = frame.robots_blue[i]
            allies.append(((lenght - robot.x) * 100,(width - robot.y) * 100))
        enemies = []
        for i in frame.robots_yellow:
            robot = frame.robots_yellow[i]
            enemies.append(((lenght - robot.x) * 100,(width - robot.y) * 100))

    #print(angle_rob)
    path = univector.update(robot_pos,ball_pos,ball_pos,math.pi,allies,enemies,index)
    #print(path)
    for i in range (len(path)):
      if yellow:
        path[i] = (path[i][0]/100 - lenght, path[i][1]/100 - width) 
      else:
        path[i] = (lenght - path[i][0]/100, width - path[i][1]/100)
    #print("NEW", path)
        


    return path
    '''obj_pos = None
    
    obj_pos = beh.decideAction(ball_pos, ball_speed, robot_pos)
  

    #
    # (obj_pos)
    ret = None
    if(obj_pos == None):
        speeds = utils.spin(robot_pos, ball_pos, ball_speed)
        ret = (speeds[0], speeds[1])

    else:
        if(goalie):
            next_step = univectorPosture.update(robot_pos,ball_pos,obj_pos,allies,enemies,index)
            speeds = p.run(angle_rob, next_step, robot_pos)
        else:
            next_step = univectorPosture.update(robot_pos,ball_pos,obj_pos,allies,enemies,index)
            speeds = p.run(angle_rob, next_step, robot_pos)
        ret = (speeds[0], speeds[1])
    
    if(yellow):
        return (ret[1], ret[0])
    else:
        return ret'''
from dataclasses import dataclass
from rc_gym.ssl.ssl_shoot_goalie.geometry import *


@dataclass
class shootGoalieState():
  """Init Frame object."""
  ball_x: float = None
  ballY: float = None
  ball_vx: float = None
  ball_vy: float = None
  robot_w: float = None
  distance: float = None
  theta_l_sen: float = None
  theta_l_cos: float = None
  theta_r_sen: float = None
  theta_r_cos: float = None
  theta_goalie_c_sen: float = None
  theta_goalie_c_cos: float = None
  theta_goalie_l_sen: float = None
  theta_goalie_l_cos: float = None
  theta_goalie_r_sen: float = None
  theta_goalie_r_cos: float = None

  def __init__(self, field_params):
    self.CENTER_GOAL_X = -field_params['field_length']/2
    self.CENTER_GOAL_Y = 0

    self.LEFT_GOAL_X = -field_params['field_length']/2
    self.LEFT_GOAL_Y = -field_params['goal_width']/2

    self.RIGHT_GOAL_X = -field_params['field_length']/2
    self.RIGHT_GOAL_Y = field_params['goal_width']/2

    self.ROBOT_RADIUS = 0.09


  def getDistance(self, frame) -> float:
    return float(mod(abs(frame.robots_blue[0].x-self.CENTER_GOAL_X), abs(frame.robots_blue[0].y-self.CENTER_GOAL_Y)))

  def getLeftPoleAngle(self, frame):
    dist_left = [frame.robots_blue[0].x - self.LEFT_GOAL_X, frame.robots_blue[0].y - self.LEFT_GOAL_Y]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robots_blue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  def getRightPoleAngle(self, frame):
    dist_right = [frame.robots_blue[0].x - self.RIGHT_GOAL_X, frame.robots_blue[0].y - self.RIGHT_GOAL_Y]
    angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robots_blue[0].theta))
    return math.sin(angle_right), math.cos(angle_right)

  def getGoalieCenterUnifiedAngle(self, frame):
    dist_g = [frame.robots_blue[0].x - frame.robots_yellow[0].x, frame.robots_blue[0].y - frame.robots_yellow[0].y]
    angle_g = toPiRange(angle(dist_g[0], dist_g[1]) + (math.pi - frame.robots_blue[0].theta))
    return angle_g
  
  def getGoalieCenterAngle(self, frame):
    angle_c = self.getGoalieCenterUnifiedAngle(frame)
    return math.sin(angle_c), math.cos(angle_c)

  def getGoalieLeftAngle(self, frame):
    dist_left = [frame.robots_blue[0].x - frame.robots_yellow[0].x, frame.robots_blue[0].y - (frame.robots_yellow[0].y - self.ROBOT_RADIUS)]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robots_blue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  def getGoalieRightAngle(self, frame):
    dist_right = [frame.robots_blue[0].x - frame.robots_yellow[0].x, frame.robots_blue[0].y - (frame.robots_yellow[0].y + self.ROBOT_RADIUS)]
    angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robots_blue[0].theta))
    return math.sin(angle_right), math.cos(angle_right)
  
  def getBallLocalCoordinates(self, frame):
    robot_ball = [frame.robots_blue[0].x - frame.ball.x, frame.robots_blue[0].y - frame.ball.y]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robots_blue[0].theta))
    robot_ball_x = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_y = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_x, robot_ball_y
  
  def getBallLocalSpeed(self, frame):
    robot_ball = [frame.robots_blue[0].v_x - frame.ball.v_x, frame.robots_blue[0].v_y - frame.ball.v_y]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robots_blue[0].theta))
    robot_ball_vx = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_vy = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_vx, robot_ball_vy

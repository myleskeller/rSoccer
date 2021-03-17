import math
import numpy as np


# Math methods
# ----------------------------
def distance(pointA, pointB):

    x1 = pointA[0]
    y1 = pointA[1]

    x2 = pointB[0]
    y2 = pointB[1]

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return distance

def mod(x, y):
    return (math.sqrt(x*x + y*y))


def angle(x, y):
    return math.atan2(y, x)


def toPositiveAngle(angle):
    return math.fmod(angle + 2 * math.pi, 2 * math.pi)


def smallestAngleDiff(target, source):
    a = toPositiveAngle(target) - toPositiveAngle(source)

    if a > math.pi:
        a = a - 2 * math.pi
    elif a < -math.pi:
        a = a + 2 * math.pi

    return a

def toPiRange(angle):
    angle = math.fmod(angle, 2 * math.pi)
    if angle < -math.pi:
        angle = angle + 2 * math.pi
    elif angle > math.pi:
        angle = angle - 2 * math.pi

    return angle
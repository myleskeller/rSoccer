m_dMin = 3.48
m_de = 7.37
m_kR = 6.15
m_delta = 4.57
m_d = 1

import deterministic_vss.utils as utils
import deterministic_vss.Field as Field
import math

def moveToGoalField(source, target, thetaDir, position):
    g_size = 2
    px = source[0]
    py = source[1]
    target = (target[0] + math.cos(thetaDir+math.pi)*m_d,target[1] + math.sin(thetaDir+math.pi)*m_d)
    gx = target[0]
    gy = target[1]
    ph_x = px * math.cos(-thetaDir) - py * math.sin(-thetaDir) - gx * math.cos(-thetaDir) + gy * math.sin(-thetaDir)
    ph_y = px * math.sin(-thetaDir) + py * math.cos(-thetaDir) - gx * math.sin(-thetaDir) - gy * math.cos(-thetaDir)
    theta_up = math.atan2((ph_y - m_de - g_size), ph_x) + thetaDir
    theta_down = math.atan2((ph_y + m_de + g_size), ph_x) + thetaDir
    rho_up = utils.mod_vec((ph_x, ph_y - m_de - g_size))
    rho_down = utils.mod_vec((ph_x, ph_y + m_de + g_size))

    TG = (-gx + (target[0] + math.cos(thetaDir)),-gy + (target[1] + math.sin(thetaDir)))
    TS = (-gx + position[0], -gy + position[1])
    dist = utils.euclideanDistance(target,position) 
    TS = (TS[0]/dist,TS[1]/dist) 


    sign = (TG[0] * TS[1] - TG[1] * TS[0])

    if (sign > 0.1):
        return theta_up + math.pi / 2. * (2 - (m_de + m_kR) / (rho_up + m_kR))
    elif(sign < -0.1):

        return theta_down - math.pi / 2. * (2 - (m_de + m_kR) / (rho_down + m_kR))
    else:
        if(utils.smallestAngleDiff(math.atan2(position[1]-gy,position[0]-gx),thetaDir) > math.pi_2):
            return thetaDir
        else:
            phi_pr = theta_up + math.pi / 2. * (2 - (m_de + m_kR) / (rho_up + m_kR))
            phi_pl = theta_down - math.pi / 2. * (2 - (m_de + m_kR) / (rho_down + m_kR))
            yl = ph_y + m_de + g_size
            yr = ph_y - m_de - g_size
            nH_pl(math.fabs(yr)*math.cos(phi_pl) / (2 * m_de), math.fabs(yr)*math.sin(phi_pl) / (2 * m_de))
            nH_pr(math.fabs(yl)*math.cos(phi_pr) / (2 * m_de), math.fabs(yl)*math.sin(phi_pr) / (2 * m_de))
            finalVector(nH_pl[0] + nH_pr[0], nH_pl[1] + nH_pr[1])
            return math.atan2(finalVector[1], finalVector[0])
    
  


def adjustToObstacle(source, direction, obstaclePosition):
    Ro = 3.0
    M = 11.0
    distance = utils.euclideanDistance(source, obstaclePosition)
    length = math.fabs((obstaclePosition[0] - source[0]) * math.sin(direction) + (source[1] - obstaclePosition[1]) * math.cos(direction))
    angle = math.atan2(obstaclePosition[1] - source[1], obstaclePosition[0] - source[0])
    diff_angle = utils.smallestAngleDiff(direction, angle)

    if (length < Ro + M and math.fabs(diff_angle) < math.pi / 2.0):
        if (distance <= Ro):
            direction = angle - math.pi
        elif (distance <= Ro + M):
            alfa = 0.6
            
            if (diff_angle > 0.):
                alfa = 1.5
            
            tmpx = ((distance - Ro) * math.cos(angle - alfa * math.pi) + (Ro + M - distance) * math.cos(angle - math.pi)) / M
            tmpy = ((distance - Ro) * math.sin(angle - alfa * math.pi) + (Ro + M - distance) * math.sin(angle - math.pi)) / M
            direction = math.atan2(tmpy, tmpx)
        else:
            multiplier = -1.0

            if (diff_angle > 0.):
                multiplier = 1.0
            

            direction = multiplier * math.fabs(atan((Ro + M) / math.sqrt(distance * distance + (Ro + M) * (Ro + M)))) + angle
        
    

    return direction


def getVectorDirection(thetaDir, canProject, position, curPos, objPos):
    initialPosition = position
    robotPosition = curPos
    ballPosition = objPos


    direction = moveToGoalField(robotPosition, ballPosition, thetaDir, initialPosition)

    '''if(self().shouldAvoidArea() or not CoachUtils::insideOurArea(self().position())){
        for (Enemy &enemy : vss.enemies()) {
        Point currentPos(enemy.position())
        direction = adjustToObstacle(robotPosition, direction, currentPos)
        }
    }'''

    return direction


def update(robotPos, ballPos, objectivePos, objectiveAngle, allies, enemies, index):

    iterations = 3
    curPos = robotPos
    angleBallGoal = objectiveAngle
    thetaDir = angleBallGoal

    canProject = True

    if(ballPos[1] > 105 or ballPos[1] < 25):
        canProject = False
    

    for i in range(iterations):
        angle = getVectorDirection(thetaDir, canProject, robotPos, curPos, objectivePos)

        curPos = (curPos[0] + math.cos(angle) * 5, curPos[1] + math.sin(angle) * 5)


    angle = getVectorDirection(thetaDir, canProject,robotPos,robotPos,objectivePos)
    
    safe = True

    for i in range(len(allies)):
        if (i == index):
            continue
        if(utils.insideOurArea(allies[i],0,0)):
            safe = False
        
        if(utils.insideEnemysArea(allies[i],0,0)):
            safe = False
        
    
    if ( not safe and (utils.insideOurArea(curPos,0,0) or utils.insideEnemysArea(curPos,0,0))):
        if(curPos[1]  < Field.goalMin[1]):
            if(robotPos[1] < Field.goalAreaMin[1]):
                curPos = (utils.bound(curPos[0], Field.offsetX+Field.goalAreaWidth+(utils.halfAxis*2.75), Field.goalAreaMin[0]-(utils.halfAxis*2.75)),curPos[1])
            else:
                curPos = (curPos[0],utils.bound(curPos[1],0,Field.goalAreaMin[1] - (utils.halfAxis*2.75)))
            
        elif(curPos[1] > Field.goalMax[1]):
            if(robotPos[1] > Field.goalAreaMax[1]):
                curPos = (utils.bound(curPos[0], Field.offsetX+Field.goalAreaWidth+(utils.halfAxis*2.75), Field.goalAreaMin[0]-(utils.halfAxis*2.75)),curPos[1])
            else:
                curPos = (curPos[0],utils.bound(curPos[1],Field.goalAreaMax[1] + (utils.halfAxis*2.75),Field.size[1]))
            
        else:
            curPos = (utils.bound(curPos[0], Field.offsetX+Field.goalAreaWidth+(utils.halfAxis*2.75), Field.goalAreaMin[0]-(utils.halfAxis*2.75)),curPos[1])
        
    
    


    curPos = (curPos[0],utils.bound(curPos[1], Field.m_min[1] + 11, Field.m_max[1] - 11)) 
  

    return curPos

  

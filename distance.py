import math

# tag height is the note height from the g

def getDistance(xReal, yReal, verticalPixelHeight, horizontalPixelWidth, tagHeight):
    x = xReal-horizontalPixelWidth/2
    y=verticalPixelHeight/2 -yReal

    diagonalFOV=(120)*(math.pi/180)
    f = math.sqrt(horizontalPixelWidth*horizontalPixelWidth+verticalPixelHeight*verticalPixelHeight)/(2*(math.tan(diagonalFOV/2)))
    # 587.4786864517579
    mountHeight=12
    mountAngle=(0)*(math.pi/180)

    VertAngle = mountAngle+math.atan(y/f)
    yDist = (tagHeight-mountHeight)/math.tan(VertAngle)
    xDist = ((tagHeight-mountHeight)/math.sin(VertAngle))*x/(math.sqrt(f*f+y*y))
    # return(xDist,yDist)

print(getDistance(378.3819580078125, 132.2773895263672, 553, 1592, 0))
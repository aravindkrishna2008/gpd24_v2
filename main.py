# imports
import cv2
import numpy as np
import math

noteName = 'testImages294'
dot = 'png'

# Load the image
image_og = cv2.imread(f'notes/{noteName}.{dot}')
image = cv2.imread(f'notes/{noteName}.{dot}')


# lower image resolution just to make sure it works with lower resolution images
scale_percent = 100
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image_og = cv2.resize(image_og, dim, interpolation = cv2.INTER_AREA)

# filter by color
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 10, 150]) # FIXME: tune for different cameras
upper = np.array([150,100,256])  # FIXME: tune for different cameras
mask = cv2.inRange(image, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite(f'{noteName}_filtered.jpeg', result)


# draw contour based on ellipse shape thing
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)

    if area > 40:
        cv2.ellipse(image_og, cv2.fitEllipse(contour), (0,255,0), 2)
        # print center of the ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        print(f'x: {x}, y: {y}')

# print height of image
print(f'height: {image_og.shape[0]}')
print(f'width: {image_og.shape[1]}')

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
    return(xDist,yDist)

print(getDistance(298.69476318359375, 256.1468505859375, 640, 480, 0))

# Save the image
cv2.imwrite(f'{noteName}_test.jpeg', image_og)

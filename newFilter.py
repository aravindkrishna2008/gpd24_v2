import cv2
import numpy as np
import time
import math

def filterImage(noteName, dot):
    image = cv2.imread(f'notes/{noteName}.{dot}')
    image_og = image.copy()
    image_og_2 = image.copy()

    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image_og = cv2.resize(image_og, dim, interpolation = cv2.INTER_AREA)

    lower = np.array([29, 30, 210]) # FIXME: tune for different cameras
    upper = np.array([150,100,256])  # FIXME: tune for different cameras
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{noteName}_filtered.jpeg', result)

    grouping = [
        # dummy value to make sure that there is no null pointer exception
        [[0,0], [[0,0]]]
    ]

    print(result[254,291])

    non_zero_pixels = np.transpose(np.nonzero(result[:,:,0]))
    print(non_zero_pixels)

    kValue = 5
    createGrouping = True
    for pixel in non_zero_pixels:
        i, j = pixel
        # print(result[i,j])
        if (result[i,j][2] > 200 and result[i,j][0] < 20):
            if createGrouping or len(grouping) == 0 or not ((np.abs(grouping[-1][0][0] - i) < kValue) or (np.abs(grouping[-1][0][1] - j) < kValue)):
                grouping.append([pixel, [pixel.tolist()]])
                createGrouping = False
            else:
                grouping[-1][1].append(pixel.tolist())
                print([int((grouping[-1][0][0] + i) / 2), int((grouping[-1][0][1] + j) / 2)])
                grouping[-1][0] = [int((grouping[-1][0][0] + i) / 2), int((grouping[-1][0][1] + j) / 2)]
        else:
            print("createGrouping = True")
            createGrouping = True
    
    upper = np.array([100,100,256])  # FIXME: tune for different cameras, should be good for the ultrawdie one
    lower = np.array([0, 0, 100]) # FIXME: tune for different cameras, should be good for the ultrawdie one
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    

    for group in grouping[1:]:
        pixels = np.array(group[1])

        maxXLeft = pixels[:,0].min()
        maxXRight = pixels[:,0].max()
        maxYTop = pixels[:,1].min()
        maxYBottom = pixels[:,1].max()

        x, y = maxYBottom, maxXLeft
        x2, y2 = maxYTop, maxXRight

        # print ratio between h and w
        print("ratio x/y", abs((x2-x) / (y2-y)))
        print("ratio y/x", abs((y2-y) / (x2-x)))
        if abs((x2-x) / (y2-y)) < 12 and abs((y2-y) / (x2-x)) < 2:
        # if True:
            cv2.rectangle(image_og, (x, y), (x2, y2), (0,255,0), 2)
            dotCoord = (int((x-x2)/2+x2),y2)
            cv2.circle(image_og_2, dotCoord, 2, (0, 255, 0),-1)

    cv2.imwrite(f'{noteName}_rect.jpeg', image_og)
    cv2.imwrite(f'{noteName}_dot.jpeg', image_og_2)
    # return dotCoord
        

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


start_time = time.time()
arr = filterImage('testImages1000', 'png')
# print(getDistance(arr[0], arr[1], 640, 480, 0))
end_time = time.time()

print(f"Processing time: {end_time - start_time} seconds")

        
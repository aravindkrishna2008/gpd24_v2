import cv2
import numpy as np

def filterImage(noteName, dot):
    # Load the image
    image_og = cv2.imread(f'notes/{noteName}.{dot}')
    image = cv2.imread(f'notes/{noteName}.{dot}')


    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image_og = cv2.resize(image_og, dim, interpolation = cv2.INTER_AREA)

    # filter by color
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([29, 30, 210]) # FIXME: tune for different cameras
    upper = np.array([150,100,256])  # FIXME: tune for different cameras
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(f'{noteName}_filtered.jpeg', result)

    # convert to hsv
    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{noteName}_filtered.jpeg', result)

    grouping = [
        [[0,0], [[0,0]]]
    ]

    kValue = 40
    # list out all hsv values of the filtered part of the image not the blakc ones
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i][j][0] != 0:
                if len(grouping) == 0 and (grouping[-1][0][0] - i < kValue and grouping[-1][0][0] - i > -kValue) or (grouping[-1][0][1] - j < kValue and grouping[-1][0][1] - j > -kValue):
                    grouping[-1][1].append([i,j])
                else:
                    grouping.append([[i,j], [[i,j]]])
    
    for i in range(1,len(grouping)):
        maxXLeft = 100000
        maxXRight = 0
        maxYTop = 100000
        maxYBottom = 0

        print(grouping[i][1])


        for j in range(len(grouping[i][1])):
            if grouping[i][1][j][0] > maxXRight:
                maxXRight = grouping[i][1][j][0]
            if grouping[i][1][j][0] < maxXLeft:
                maxXLeft = grouping[i][1][j][0]
            if grouping[i][1][j][1] > maxYBottom:
                maxYBottom = grouping[i][1][j][1]
            if grouping[i][1][j][1] < maxYTop:
                maxYTop = grouping[i][1][j][1]

        print(maxXLeft, maxXRight, maxYTop, maxYBottom)
        cv2.rectangle(result, (maxXLeft, maxYTop), (maxXRight, maxYBottom), (0,255,0), 2)

        # maxYBottom, maxXLeft
        # maxYTop, maxXRight
        x, y = maxYBottom, maxXLeft
        x2, y2 = maxYTop, maxXRight

        cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        cv2.circle(result, (x2, y2), 2, (0, 255, 0), -1)

        cv2.rectangle(image_og, (x, y), (x2, y2), (0,255,0), 2)

        cv2.imwrite(f'{noteName}_filtered.jpeg', image_og)

filterImage('testImages161', 'png')

        
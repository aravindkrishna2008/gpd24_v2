# imports
import cv2
import numpy as np

noteName = '2notes'
dot = 'png'

# Load the image
image_og = cv2.imread(f'notes/{noteName}.{dot}')
image = cv2.imread(f'notes/{noteName}.{dot}')



# lower image resolution just to make sure it works with lower resolution images
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image_og = cv2.resize(image_og, dim, interpolation = cv2.INTER_AREA)

# filter by color
# image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([30, 90, 210]) # FIXME: tune for different cameras
upper = np.array([70,160,256])  # FIXME: tune for different cameras
mask = cv2.inRange(image, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite(f'notes/{noteName}_filtered.jpeg', result)


# draw contour based on ellipse shape thing
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 200:
        cv2.ellipse(image_og, cv2.fitEllipse(contour), (0,255,0), 2)

# Save the image
cv2.imwrite(f'notes/{noteName}_test.jpeg', image_og)

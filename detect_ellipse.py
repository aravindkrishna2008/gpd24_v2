import cv2
from matplotlib.pyplot import draw
import numpy as np
import time
prev_large = []
prev_image = []
prev_llpy = []
def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Assumes the ellipse is 'axis aligned' - modelled by setting the cross product term equal to 1

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    """


    #Quadratic terms of the design matrix
    D1 = np.vstack([x**2, np.ones(len(x)), y**2]).T
    # Linear terms of the design matrix
    D2 = np.vstack([x, y, np.ones(len(x))]).T

    #Sections of the scatter matrix
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigenval, eigenvec = np.linalg.eig(M)
    con = 4 * eigenvec[0]* eigenvec[2] - eigenvec[1]**2
    ak = eigenvec[:, np.nonzero(con > 0)[0]]

    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the Cartesian coefficients, (a, b, c, d, e, f), to polar parameters, 
    x0, y0, ap, bp, e, phi, where 
    (x0, y0) is the ellipse centre; 
    (ap, bp) are the semi-major and semi-minor axes, respectively; 
    e is the eccentricity; and 
    phi is the rotation of the semi-major axis from the x-axis.

    """

    if(len(coeffs)==0): 
        return

    # assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    det = b**2 - a*c
    # if det > 0:
    #     raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
    #                      ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / det, (a*f - b*d) / det

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / det / (fac - a - c))
    bp = np.sqrt(num / det / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    # phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """
    if params is None: return

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

# convert from pixels to angles
# screen width (px) = 320
# screen height (px) = 240
# screen FOV x (deg) = 59.6
# screen FOV y (deg) = 49.7
def px_to_deg(cx, cy):
    tx = ((cx - 480.0) / 960.0) * 59.6
    ty = ((cy - 360.0) / 720.0) * 49.7
    return tx, -ty

def draw_point(image, x, y):
    cv2.circle(image, (int(x), int(y)), 2, (0,0,255), cv2.FILLED)

def draw_point_2(image, x, y):
    cv2.circle(image, (int(x), int(y)), 2, (0,255, 0), cv2.FILLED)

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    # start = time.time()
    try:
        # convert the input image to the HSV color space
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_threshold= cv2.inRange(img_hsv,(30, 40, 118), (85, 255, 255))
        img_threshold = cv2.medianBlur(img_threshold, 5)
        # img_threshold = cv2.blur(img_threshold, 1)
        img_threshold= cv2.inRange(img_threshold, 150, 255)
        cv2.imshow("img", img_threshold)
        # h, s, v = cv2.split(image)
        # img_threshold = cv2.inRange(v, 200, 255)
        
        # find contours in the new binary image
        contours, _ = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_threshold, contours, -1, 100, 2)

        points = []
        llpython = []
        largestContour = [[]]
        # largestContour = max(contours, key=cv2.contourArea) # required to be returned by api
        # for i in range(len(contours)):
        for coord in contours[0]:
                    points.append(coord[0])


        # if(len(points) >= 5):
        x = []
        y= []
        for point in points:
            if(point[0] != 0 and point[1]!=0):
                x.append(point[0])
                y.append(point[1])
        x=np.array(x)
        y = np.array(y)
        e = fit_ellipse(x, y)
        e2 = cv2.fitEllipse(contours[0])
        params = cart_to_pol(e)
        if params is None: return
        x, y = get_ellipse_pts(params)
        for i in range(len(x)):
            draw_point_2(image, x[i], y[i])
        max = 9999
        for i in range(len(y)):
            if y[i] < max: max = y[i]
        

        # x_top, y_top = x[index], y[index]
        # draw_point(image, x[index], y[index])
        # draw_point(image, int(params[0]), int(params[1]-params[3]))

        tx, ty = px_to_deg(abs(params[0]), abs(max))

        # initialize an array of values to send back to the robot
        llpython = [1, tx, ty, 0, 0, 0]

        #return the largest contour for the LL crosshair, the modified image, and custom robot data
        cv2.imshow("ret", image)
        end = time.time()
        print("you: " + str(t2-t1) + "\n OpenCV: "+ str(t3-t2))
        return largestContour, image, llpython
    except:
        print("error")
        return [[]], image, [-1, 0, 0, 0, ]

# img = cv2.imread('img3.png')
# runPipeline(img, llrobot=[60, 70, 70, 95, 255, 255])
# cv2.imshow('image',img)
# cv2.waitKey(0)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)
# output = cv2.VideoWriter('vid.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, (640, 360), 0)
   
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
# Read until video is completed
i = 1
while(cap.isOpened()):
  if i%2==0: continue
      
  # Capture frame-by-frame
  ret, frame = cap.read()

  frame = frame[50:, 50:]
  if ret == True:
   
    # Display the resulting frame
    try:
        st = time.time()
        _, img, llpython = runPipeline(frame, llrobot=[40, 20, 145, 255, 190, 255])
        end = time.time()
        print("total: " + str(end-st))
        cv2.imshow("ret", img)
        # output.write(frame)
    except:
        continue
   
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #   break
   
  # Break the loop
  else: 
    break
   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()
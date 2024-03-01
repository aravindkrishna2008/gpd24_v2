import cv2

camera = cv2.VideoCapture(0)

c = 500

# image 2 75 in and 14 in

while True:
    ret, frame = camera.read()
    cv2.imshow('frame', frame)
    cv2.imwrite(f'notes/testImages{c}.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # c+=1

camera.release()
cv2.destroyAllWindows()
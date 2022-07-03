import numpy as np
import cv2

# Models
haarcascade = "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(haarcascade)

LBFmodel = "lbfmodel.yml"
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def fat_circ(image, landmark):
    cv2.circle(image, (int(landmark[0]), int(landmark[1])), 2, (255, 255, 255), 5)
def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar("smoothness", "image" , 8, 10, nothing)   

x_min = 0
x_max = 1000
y_min = 0
y_max = 1000
pct_x = 0
pct_y = 0
c = 0
click = False

cursor_x = 0
cursor_y = 0

screen_w = 1280
screen_h = 720
offset = 0
smoothness = 8

click_dist = 1000

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.flip(image, 1)


    # Find landmark
    faces = detector.detectMultiScale3(image, 1.1, 3, 0, (200, 200), (1000, 1000), True)
    for idx in range(len(faces[0])):
        (x, y, w, d) = faces[0][idx]
        # cv2.rectangle(image, (x, y), (x+w, y+d), (255, 255, 255), 5)
        # cv2.putText(image, f"{faces[2][idx] / 100:.2%}", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, 2)

    # Find anchors
    if len(faces[0]) > 0:
        _, landmarks = landmark_detector.fit(image, faces[0])
        for landmark in landmarks:
            # for x,y in landmark[0]:
            #     cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 3)

            # cv2.circle(image, (int(landmark[0][30][0]), int(landmark[0][30][1])), 2, (255, 255, 255), 5)

            anchor_C = landmark[0][30]
            fat_circ(image, anchor_C)

            anchor_L = landmark[0][1]
            fat_circ(image, anchor_L)

            anchor_R = landmark[0][15]
            fat_circ(image, anchor_R)

            anchor_D = landmark[0][8]
            fat_circ(image, anchor_D)

            anchor_LU = landmark[0][57]
            fat_circ(image, anchor_LU)

            anchor_LB = landmark[0][51]
            fat_circ(image, anchor_LB)

            y = ((anchor_L[1] + anchor_R[1]) / 2) - anchor_D[1]
            x = ((anchor_C[0] + anchor_D[0]) / 2) - ((anchor_L[0] + anchor_R[0]) / 2)

            c = anchor_LU[1] - anchor_LB[1]

            pct_x = (x - x_min) / (x_max - x_min)
            pct_y = (y - y_min) / (y_max - y_min)
            # print(f"x:{x:.1f} y:{y:.1f}")

            cv2.putText(image, f"x:{pct_x:.1f} y:{pct_y:.1f}", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, 2)

            break

    

    # cv2.rectangle(image, (offset, offset), (screen_w, screen_h), (255, 255, 255), 4)
    # cv2.circle(image, (int(offset + (screen_w - offset) * pct_x), int(offset + (screen_h - offset) * pct_y)), 1, (255, 255, 255), 4)

    # Cursor
    smoothness = cv2.getTrackbarPos("smoothness", "image")
    cursor_x += (pct_x - cursor_x) / smoothness
    cursor_y += (pct_y - cursor_y) / smoothness
    cv2.circle(image, (int(offset + (screen_w - offset) * cursor_x), int(offset + (screen_h - offset) * cursor_y)), 1, (255, 255, 255), 4)

    # Click
    click = c > click_dist
    cv2.putText(image, "CLICK" if click else "NOT CLICK", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, 2)


    # Display the resulting frame
    cv2.imshow('image', image)

    user_input = cv2.waitKey(1)
    if user_input == ord('q'):
        break
    if user_input == ord('1'): # min
        x_min = x
        print(f"x min = {x_min}")
    if user_input == ord('2'): # max
        x_max = x
        print(f"x max = {x_max}")
    if user_input == ord('3'): # min
        y_min = y
    if user_input == ord('4'): # max
        y_max = y
    if user_input == ord('5'): # max
        click_dist = c

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import os


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

gestureList = []
folderPath = 'FingerImages'
imgList = os.listdir(folderPath)
for imgPath in imgList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    gestureList.append(image)

tipID = [4, 8, 12, 16, 20]


def gestureRecognize(fingers, img):
    count = fingers.count(1)
    overlayImage = []
    if count == 5:
        overlayImage = gestureList[2]
    elif count == 0 or (count == 1 and fingers[0] == 1):
        overlayImage = gestureList[0]
    elif count == 2 and fingers[1] == 1 and fingers[2] == 1:
        overlayImage = gestureList[1]
    else:
        overlayImage = []
    if len(overlayImage) != 0:
        h, w, c = overlayImage.shape
        img[0:h, 0:w] = overlayImage


def handRecognize(lmList, img):
    if len(lmList) != 0:
        if lmList[tipID[0]][0] > lmList[tipID[4]][0]:  # rightHand
            rightHandRecognize(lmList, img)
        elif lmList[tipID[0]][0] < lmList[tipID[4]][0]:
            leftHandRecognize(lmList, img)


def rightHandRecognize(lmList, img):
    fingers = []
    # 大拇指
    if lmList[tipID[0]][0] > lmList[tipID[0]-1][0]:  # finger open:1 close:0
        fingers.append(1)
    else:
        fingers.append(0)

    # 其他四指
    for id in range(1, 5):
        if lmList[tipID[id]][1] < lmList[tipID[id]-2][1]:  # finger open:1 close:0
            fingers.append(1)
        else:
            fingers.append(0)

    gestureRecognize(fingers, img)


def leftHandRecognize(lmList, img):
    fingers = []
    # 大拇指
    if lmList[tipID[0]][0] < lmList[tipID[0]-1][0]:  # finger open:1 close:0
        fingers.append(1)
    else:
        fingers.append(0)

    # 其他四指
    for id in range(1, 5):
        if lmList[tipID[id]][1] < lmList[tipID[id]-2][1]:  # finger open:1 close:0
            fingers.append(1)
        else:
            fingers.append(0)

    gestureRecognize(fingers, img)


def main():
    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)
            height, width, c = img.shape

            lmList = []

            handLms = result.multi_hand_landmarks
            if handLms:
                for hLms in handLms:
                    mpDraw.draw_landmarks(img, hLms, mpHands.HAND_CONNECTIONS)
                    for i, lm in enumerate(hLms.landmark):  # getHandsPosition
                        xP = lm.x*width
                        yP = lm.y*height
                        lmList.append([xP, yP])
            handRecognize(lmList, img)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()

# GestureRecognition
猜拳的手勢辨識，使用的套件為OpenCV+MediaPipe
## Demo
![image](https://raw.githubusercontent.com/johnny990628/GestureRecognition/master/img_2022-01-18_23-16-16_AdobeCreativeCloudExpress.gif)

## Code
```
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
```
使用OpenCV開啟並讀取視訊鏡頭，因為OpenCV讀出來的圖片顏色模型為BGR，而MediaPipe辨識的顏色模型為RGB，因此在處理圖片時會先轉為RGB來進行處理。
接著利用MediaPipe內建的Function(multi_hand_landmarks)來進行手部節點的標記，最後再把節點間連結起來，取得手部於畫面的座標。
![image](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

```
def handRecognize(lmList, img):
    if len(lmList) != 0:
        if lmList[tipID[0]][0] > lmList[tipID[4]][0]:  # rightHand
            rightHandRecognize(lmList, img)
        elif lmList[tipID[0]][0] < lmList[tipID[4]][0]: #leftHand
            leftHandRecognize(lmList, img)
```
利用了剛剛存取手部座標的List，來辨識舉起的手為右手還左手，其原理是判斷大拇指的X座標在中指的左還是右。

```
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

```
利用張開的手指數來進行手勢的判斷，張開為1閉合為0，其判斷為指尖與關節的Y座標。

```
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
```
判斷張開的手指數顯示相對應的圖片。



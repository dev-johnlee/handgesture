import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#solutions의 drawing_utils은 landmark를 그리는 도구
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    #success 은 카메라 상태 이며, 정상 : True, 비정상 : False
    #img 은 현재시점의 플레임
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx,cy)
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    # 프레임 계산
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # print("Estimated fps {0} ".format(fps))


    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255),3)

    """
    [landmark {
      x: 0.5801884531974792
      y: 0.34119316935539246
      z: -3.548683162080124e-05
    }
    """

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break;
#
#
# def main():
#     pTime = 0
#     cTime = 0
#
#     while True:
#         # success 은 카메라 상태 이며, 정상 : True, 비정상 : False
#         # img 은 현재시점의 플레임
#         success, img = cap.read()
#
#
# if __name__ == "__main__":
#     main()

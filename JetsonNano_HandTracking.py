import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

#
# #순서 : 엄지,검지,중지,약지,소지
# compareIndex = [[1,4],[6,8],[10,12],[14,16],[18,20]]
#
# open = [False, False, False, False, False]
#
# #순서 : 엄지,검지,중지,약지,소지,제스쳐이름
# gesture = [[True, True, True, True, True, "Call to Emergency center"],
#             [False, True, True, False, False, "Turn on the light"]
#             [True, True, False, False, True, "Open the door"]]



pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = htm.handDetector()
while True:
    # success 은 카메라 상태 이며, 정상 : True, 비정상 : False
    # img 은 현재시점의 플레임
    success, img = cap.read()
    img = detector.findHands(img=img)
    lmList = detector.findPosition(img=img)
    #draw=False
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    # 프레임 계산
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break;
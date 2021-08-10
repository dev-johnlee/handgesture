import cv2
import mediapipe as mp
import time
import keyboard
import numpy as np
# import csv
import pandas as pd


max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five'
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

df = pd.read_csv('data/train.csv', names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks is not None:
        for handLms in results.multi_hand_landmarks:

            #각 랜드마크 = joint ((21개 joint, 3개의 좌표*x,y,z))
            joint = np.zeros((21,3))
            for id, lm in enumerate(handLms.landmark):
                joint[id] = [lm.x, lm.y, lm.z]
            # 관절별로 Vector를 구함
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] #parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] #child joint

            v = v2-v1 #[20,3]
            v=v/np.linalg.norm(v, axis=1)[:, np.newaxis]

            comparedV1 = v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:]
            comparedV2 = v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]
            angle = np.arccos(np.einsum('nt,nt->n',comparedV1,comparedV2))
            #covert radian to degree
            angle = np.degrees(angle) #각
            # print(angle)

            if keyboard.is_pressed('a'):
                numList = []
                for num in angle:
                    num = round(num, 6)
                    numList.append(str(num))
                numList.append('1')
                new_df = pd.DataFrame(numList)
                final_df = pd.concat([df, new_df], ignore_index=True)
                print(final_df)
                print("The angle data of the gesture has been saved.")
                # df.to_csv("sample.csv", index=False)
                # print(df)

            # data = np.array([angle], dtype=np.float32)
            # data = np.append(data, 11)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if keyboard.is_pressed('b'):
        break
final_df.to_csv('new_train.csv', index=False, header=False)
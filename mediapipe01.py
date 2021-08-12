import cv2
import mediapipe
import time

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

with handsModule.Hands(static_image_mode=True) as hands:
    # Hands key points detection


image = cv2.imread("C:/Users/N/Desktop/hand.jpg")
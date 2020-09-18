import numpy as np
import cv2
import dlib
import pyttsx3
import random
from playsound import playsound
from threading import Thread
import threading
import time
global isPlaying


class main(Thread):
    def run(self):
        global isPlaying
        isPlaying = False

        # Capture Video
        capt = cv2.VideoCapture(0)

        # Initialize detectors
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while True:
            ret, frame = capt.read()

            # Convert feed to grayscale and detect
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(grayscale)

            if len(faces) > 0:
                # The face landmarks code begins from here
                x1 = faces[0].left()
                y1 = faces[0].top()
                x2 = faces[0].right()
                y2 = faces[0].bottom()
                play().start()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (64,224,208), 2)
                cv2.putText(frame, "No Mask Detected", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 224, 208), 2)

            cv2.imshow('', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capt.release()
        cv2.destroyAllWindows()


class play(Thread):
    def run(self):
        global isPlaying
        if not isPlaying:
            isPlaying = True
            file = random.randrange(1, 11)
            i = file
            while file == i:
                file = random.randrange(1, 11)
            playsound("{}.mp3".format(file), False)
            time.sleep(5)
            isPlaying = False


main().start()

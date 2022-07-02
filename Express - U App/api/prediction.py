import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from segmentation import *
from model import *


actions = np.array(['hello','thanks','iloveyou'])

no_of_sequences = 30
sequence_length = 30

sequence = []
sentence = []
predictions = []
threshold = 0.8

def prediction(frame):
  with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    image,results = mediapipe_detection(frame,holistic)

    draw_styled_landmarks(image,results)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
      res = model.predict(np.expand_dims(sequence,axis=0))[0]
      # print(actions[np.argmax(res)])
      predictions.append(np.argmax(res))

      if (np.unique(predictions[-10:])[0] == np.argmax(res)
          and res[np.argmax(res)] > threshold):
        if len(sentence) > 0:
          if actions[np.argmax(res)] != sentence[-1]:
            sentence.append(actions[np.argmax(res)])
        else:
            sentence.append(actions[np.argmax(res)])

      if len(sentence) > 5:
        sentence = sentence[-5:]
  return sentence
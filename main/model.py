import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = 'E:\Final Year Project\Express-U\data\\vid_dataset'

actions = np.array(['hello','thanks','iloveyou'])

no_of_sequences = 30

sequence_length = 30

label_map = {label:num for num,label in enumerate(actions)}

sequences,labels = [],[]

for action in actions:
  for sequence in range(no_of_sequences):
    window = []
    for frame_num in range(sequence_length):
      res = np.load(os.path.join(DATA_PATH,action,str(sequence),f'{frame_num}.npy'))
      window.append(res)
    sequences.append(window)
    labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

#spliting Data
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.05)

log_dir = 'E:\Final Year Project\Express-U\data\Logs'
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])

# model.fit(X_train,y_train,epochs=2000,callbacks=[tb_callback])


from sklearn.metrics import multilabel_confusion_matrix,accuracy_score

yhat = model.predict(X_test)

ytrue = np.argmax(y_test,axis=1).tolist()
yhat = np.argmax(yhat,axis=1).tolist()

multilabel_confusion_matrix(ytrue,yhat)

accuracy_score(ytrue,yhat)

# model.save('actions.h5')
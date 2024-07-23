from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.models import Model, load_model
import cv2
import numpy as np
from os.path import abspath, join
import imutils

def model(learning_rate = 0.0001, decay = 0.00001, frames = 8):
    optimizer = Adam(lr = learning_rate, decay = decay)
    
    x = Input(shape = (frames, 160, 160, 3))

    model_path = abspath("./detectors/facenet")
    cnn = load_model(join(model_path, 'facenet_keras.h5'))
    # cnn.load_weights('weights/facenet_keras_weights.h5')
    # for layer in cnn.layers:
    #     layer.trainable=False
    x_ = TimeDistributed(cnn)(x)

    encoded_video = LSTM(256)(x_)
    fc = Dense(512)(encoded_video)
    fc = Dropout(0.5)(fc)
    out = Dense(4, activation='softmax')(fc)

    model = Model(inputs = [x], outputs = out)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.load_weights(join(model_path, 'weights/fns_t.15-0.95.hdf5'))
    return model


def normalize(img):
    img = (1/255)*img
    img[:,:,0] -= np.mean(img[:,:,0])
    img[:,:,1] -= np.mean(img[:,:,1])
    img[:,:,2] -= np.mean(img[:,:,2])
    return img


def transform_data(data):
    data = [imutils.resize(i, width=160) for i in data]
    data = data[: len(data) - (len(data) % 8)]
    data = np.asarray([normalize(img) for img in data])
    return np.reshape(data, (int(len(data)/8), 8, data.shape[1], data.shape[2], data.shape[3]))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict(data, temporal=model()):
    test = transform_data(data)
    prediction = [(1 - pred) for pred in temporal.predict(test)[:,0]]
    del temporal
    prediction = np.array(prediction, dtype=np.float32)
    return list(prediction.astype(float))
import tensorflow as tf
import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'train')
IMG_DIR = os.path.join(os.path.dirname(__file__), r'static\images')
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogsvscats-{}-{}'.format(LR, '6conv-basic-video')


def label_image(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_image(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                         (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_given_img(img):
    path = os.path.join(IMG_DIR, img)
    img_name = img.split('.')[0]
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                     (IMG_SIZE, IMG_SIZE))
    return [[np.array(img), img_name]]


def nn():
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet,
                         optimizer='adam',
                         learning_rate=LR,
                         loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model


def train_nn(model):
    train_data = create_train_data()
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    train = train_data[:-500]
    test = train_data[-500:]
    X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = [i[1] for i in train]

    X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = [i[1] for i in test]

    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)


def predict_label(img):
    if not os.path.exists('{}.meta'.format(MODEL_NAME)):
        return 'cat or dog (Train neural network first)'
    else:
        model = nn()
        model.load(MODEL_NAME)
        processed_img = process_given_img(img)
        # cat: [1, 0]
        # dog: [0, 1]
        img_data = processed_img[0][0]
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        return str_label


if __name__ == '__main__':
    model = nn()
    train_nn(model)

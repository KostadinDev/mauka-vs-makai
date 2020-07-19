from __future__ import print_function

import glob
import random

import matplotlib.pyplot as plt
import numpy as np
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import image


def loadData(class0Dirs, class1Dirs, featureExtractor):
    X1s = []
    for d in class0Dirs:
        X1s.extend(loadImagesFromDir(d, featureExtractor))
    Y1s = [0] * len(X1s)

    X2s = []
    for d in class1Dirs:
        X2s.extend(loadImagesFromDir(d, featureExtractor))
    Y2s = [1] * len(X2s)

    print("MostCommonLabelLearner accuracy", (max(len(Y2s), len(Y1s)) / (len(Y2s) + len(Y1s))))

    X1s.extend(X2s)
    Y1s.extend(Y2s)
    data = list(zip(X1s, Y1s))
    random.shuffle(data)
    lastInd = int(0.7 * len(data))
    trainData = data[:lastInd]
    trainX, trainY = zip(*trainData)
    testData = data[lastInd:]
    testX, testY = zip(*testData)

    (x_train, x_train_orig) = zip(*trainX)
    (x_test, x_test_orig) = zip(*testX)

    return ((np.array(x_train), x_train_orig, np.array(trainY)), (np.array(x_test), x_test_orig, np.array(testY)))


def loadImagesFromDir(directoryname, featureExtractor):
    images = []
    c = 0
    for f in glob.glob(directoryname + "/*.jpg"):
        c += 1
        if c % 100 == 0:
            print("loading", c)

        img = image.load_img(f, target_size=(160, 160))
        processed_img = None
        if featureExtractor is None:
            processed_img = image.img_to_array(img)
        else:
            processed_img = featureExtractor.predict(np.array([image.img_to_array(img), ]))[0]
            # TODO: Replace the above line with something that passes it through the featureExtractor

        images.append((processed_img, img))
    return images


batch_size = 128
epochs = 20

# x_train is the training data that will be fed into the network (may not be images if using a feature extractor)
# x_train_orig is the original images (could be the same as x_train if not using a feature extractor)
# y_train is the

conv_base = ResNet50(weights='imagenet',
                     include_top=False,
                     input_shape=(160, 160, 3))

(x_train, x_train_orig, y_train), (x_test, x_test_orig, y_test) = loadData(['218699362', 'hawaiivolcanoesnps'],
                                                                           ['734013394', '2186010331721786'], conv_base)
# 76800 because the dimensions are (160,160,3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Flatten(input_shape=(5, 5, 2048)))
model.add(Dense(512, activation='relu', input_shape=(76800,)))
model.add(Dense(256, activation='relu', input_shape=(76800,)))
model.add(Dense(1, activation="sigmoid", input_shape=(76800,)))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

shown = 0
for i in range(len(x_test_orig)):
    if shown >= 3:
        break  # stop after 3

    x = x_test[i]  # instance (preprocessed features, x_test_orig[i] is the ACTUAL image)
    trueVal = y_test[i]  # true label
    predictedVal = 0
    predictedProb = model.predict(np.array([x, ]))[0]  # predicted probability of label 1
    if predictedProb > 0.5:
        predictedVal = 1
    else:
        predictedVal = 0
    if predictedVal != trueVal:
        plt.imshow(x_test_orig[i])
        plt.show()
        shown += 1
    # TODO: If label doesn't match, show image and increase shown counter.
    # To get label from predictedProb, use a threshold of 0.5 (similar idea to Program 3)
    # Recall from CS172 that plt.imshow(image) followed by plt.show() will display an image.
imgs = loadImagesFromDir("images", conv_base)
x_ftest, x_ftest_orig = zip(*imgs)
for i in range(len(x_ftest)):
    x = x_ftest[i]  # instance (preprocessed features, x_test_orig[i] is the ACTUAL image)
    predictedProb = model.predict(np.array([x, ]))[0]  # predicted probability of label 1
    if predictedProb > 0.5:
        predictedVal = 1
    else:
        predictedVal = 0
    print("Probability ", predictedProb, "Label:", predictedVal)
    plt.imshow(x_ftest_orig[i])
    plt.show()
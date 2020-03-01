from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
#import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import cv2
import timeit
from Misc.TFSpatialTransformer import batch_transformer
from Misc import DLTMatrix
import math

#**************************** Data Generation ****************************

import glob

MAX_PERTURBATION = 32
BATCH_SIZE = 128

def readImgAndGeneData(inputPath, patchLength = 128, perturbation = MAX_PERTURBATION, numPairsInOneImg = 8, showImg = True):
    fileNames = glob.glob(inputPath + "/*.jpg")
    data = []
    label = []
    C_a_list = []
    H_list = []
    imgList = []
    rectList = []
    for imgIdx in range(len(fileNames)):
        #print(imgIdx)
        img_color = cv2.imread(fileNames[imgIdx])
        imgList.append(img_color)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        height = img.shape[0]
        width = img.shape[1]

        p1 = perturbation
        border = patchLength/2 + p1
        while (border * 2 >= height-4) or (border * 2 >= width-4):
            p1 = p1 / 2
            border = int(patchLength/2 + p1)

        # top, bottom, left, right border
        patchCenterLegalArea = [border+1, height-2-border, border+1, width-2-border]

        for rp in range(numPairsInOneImg):
        # random patch center
            cy = np.random.randint(patchCenterLegalArea[0], patchCenterLegalArea[1])
            cx = np.random.randint(patchCenterLegalArea[2], patchCenterLegalArea[3])

            # top left
            tlx = int(cx - patchLength/2)
            tly = int(cy - patchLength/2)

            patchA = img[tly:tly+patchLength, tlx:tlx+patchLength]
            rectList.append([tly,tly+patchLength, tlx,tlx+patchLength])

            H_4pt = np.random.randint(-p1, p1, 8)
            #H_4pt = np.zeros(8)

            #image = cv2.rectangle(img, (tlx, tly), (tlx+patchLength, tly+patchLength), (255, 0, 0) , 2)
            
            C_a = [(tlx, tly),
                   (tlx+patchLength-1, tly),
                   (tlx+patchLength-1, tly+patchLength-1),
                   (tlx, tly+patchLength-1)
                ]
            C_a_list.append(np.array(C_a).reshape(8))

            C_b = [(tlx+H_4pt[0], tly+H_4pt[1]),
                   (tlx+patchLength-1+H_4pt[2], tly+H_4pt[3]),
                   (tlx+patchLength-1+H_4pt[4], tly+patchLength-1+H_4pt[5]),
                   (tlx+H_4pt[6], tly+patchLength-1+H_4pt[7])
                ]

            if showImg:
                for i in range(4):
                    img_color = cv2.line(img_color, C_a[i], C_a[(i+1)%4], (255, 0, 0), 2)
                    img_color = cv2.line(img_color, C_b[i], C_b[(i+1)%4], (0, 255, 0), 2)
                cv2.imshow('img_color', img_color)
                cv2.imwrite('img_color.jpg', img_color)
                cv2.waitKey(0)

            H = cv2.getPerspectiveTransform(np.array(C_b, dtype = "float32"), np.array(C_a, dtype = "float32"))
            #H_list.append([[1,0,0],[0,1,0],[0,0,1]])
            #H_list.append(np.linalg.inv(H))
            warped = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
            patchB = warped[tly:tly+patchLength, tlx:tlx+patchLength]
            
            '''
            H1 = np.array([[1., 0., tlx], [0., 1., tly], [0., 0., 1]])
            H2 = np.array([[1., 0., -tlx], [0., 1., -tly], [0., 0., 1]])
            H_tot = np.matmul(H2, np.matmul(H, H1))
            '''

            tlx=0
            tly=0
            C_a_ = [(tlx, tly),
                   (tlx+patchLength-1, tly),
                   (tlx+patchLength-1, tly+patchLength-1),
                   (tlx, tly+patchLength-1)
                ]

            C_b_ = [(tlx+H_4pt[0], tly+H_4pt[1]),
                   (tlx+patchLength-1+H_4pt[2], tly+H_4pt[3]),
                   (tlx+patchLength-1+H_4pt[4], tly+patchLength-1+H_4pt[5]),
                   (tlx+H_4pt[6], tly+patchLength-1+H_4pt[7])
                ]
            H_ = cv2.getPerspectiveTransform(np.array(C_a_, dtype = "float32"), np.array(C_b_, dtype = "float32"))
            
            H_list.append(H_)

            '''
            warpedB = cv2.warpPerspective(patchA, H_tot, (patchA.shape[1], patchA.shape[0]))
            cv2.imshow('patchB', patchB)
            cv2.imshow('warpedB', warpedB)
            cv2.waitKey(0)
            '''

            mergedPatch = np.dstack((patchA, patchB))
            data.append(mergedPatch)
            label.append(H_4pt)

            if showImg:
                cv2.imshow("img", img)
                cv2.imshow("warped", warped)
                cv2.imshow("patchA", patchA)
                cv2.imshow("patchB", patchB)
                cv2.waitKey(0)


    return np.array(data).astype('float32'), np.array(label).astype('float32'), np.array(C_a_list).astype('float32').reshape([-1,1,8]), np.array(H_list).astype('float32'), imgList, rectList

#*************************************************************************



#************************** Supervised Approach **************************

def getSupervisedModel(inputShape=(128, 128, 2)):
    
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=inputShape, padding='same', name="CV1_3x3_64"))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV2_3x3_64"))
    model.add(layers.MaxPooling2D((2, 2), name="MP1_2x2"))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV3_3x3_64"))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV4_3x3_64"))
    model.add(layers.MaxPooling2D((2, 2), name="MP2_2x2"))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV5_3x3_128"))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV6_3x3_128"))
    model.add(layers.MaxPooling2D((2, 2), name="MP3_2x2"))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV7_3x3_128"))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV8_3x3_128"))

    model.add(layers.Flatten(name="Flatten"))
    model.add(layers.Dense(1024, activation='relu', name="Dense_1024"))
    model.add(layers.Dense(8, activation='tanh', name="Dense_8"))
    
    optimizer = tf.keras.optimizers.Adam(lr = 0.0001, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MSE)

    return model

#*************************************************************************



#************************* Unsupervised Approach *************************

class TensorDLT(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(TensorDLT, self).__init__()
        
    '''
    def get_config(self):
        config = {'name': self.name}
        base_config = super(MyMeanPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_config(self):
        config = super().get_config()
        config['name'] = self.name# say self._pool_size  if you store the argument in __init__
        return config
    '''

    def call(self, inputs):
        [offset, C_a] = inputs
        offset = offset*MAX_PERTURBATION
        batch_size = tf.shape(offset)[0] #BATCH_SIZE
        #C_a = tf.expand_dims(C_a, [1])
        C_a_zero_offset = tf.constant([[[0,0,127,0,127,127,0,127]]], tf.float32)
        C_a_zero_offset = tf.tile(C_a_zero_offset,[batch_size,1,1])
        offset = tf.expand_dims(offset, [1])
        C_src = tf.transpose(C_a_zero_offset, [0,2,1])
        C_dst = tf.transpose(tf.add(C_a_zero_offset, offset), [0,2,1])

        Acol0param = tf.constant(DLTMatrix.Acol0, tf.float32)
        Acol0param = tf.expand_dims(Acol0param, [0])
        Acol0param = tf.tile(Acol0param,[batch_size,1,1])

        Acol1param = tf.constant(DLTMatrix.Acol1, tf.float32)
        Acol1param = tf.expand_dims(Acol1param, [0])
        Acol1param = tf.tile(Acol1param,[batch_size,1,1])

        Acol2 = tf.constant(DLTMatrix.Acol2, tf.float32)
        Acol2 = tf.expand_dims(Acol2, [0])
        Acol2 = tf.tile(Acol2,[batch_size,1,1])

        Acol3param = tf.constant(DLTMatrix.Acol3, tf.float32)
        Acol3param = tf.expand_dims(Acol3param, [0])
        Acol3param = tf.tile(Acol3param,[batch_size,1,1])

        Acol4param = tf.constant(DLTMatrix.Acol4, tf.float32)
        Acol4param = tf.expand_dims(Acol4param, [0])
        Acol4param = tf.tile(Acol4param,[batch_size,1,1])

        Acol5 = tf.constant(DLTMatrix.Acol5, tf.float32)
        Acol5 = tf.expand_dims(Acol5, [0])
        Acol5 = tf.tile(Acol5,[batch_size,1,1])

        Acol61param = tf.constant(DLTMatrix.Acol61, tf.float32)
        Acol61param = tf.expand_dims(Acol61param, [0])
        Acol61param = tf.tile(Acol61param,[batch_size,1,1])

        Acol62param = tf.constant(DLTMatrix.Acol62, tf.float32)
        Acol62param = tf.expand_dims(Acol62param, [0])
        Acol62param = tf.tile(Acol62param,[batch_size,1,1])

        Acol71param = tf.constant(DLTMatrix.Acol71, tf.float32)
        Acol71param = tf.expand_dims(Acol71param, [0])
        Acol71param = tf.tile(Acol71param,[batch_size,1,1])

        Acol72param = tf.constant(DLTMatrix.Acol72, tf.float32)
        Acol72param = tf.expand_dims(Acol72param, [0])
        Acol72param = tf.tile(Acol72param,[batch_size,1,1])

        bcol0param = tf.constant(DLTMatrix.bcol0, tf.float32)
        bcol0param = tf.expand_dims(bcol0param, [0])
        bcol0param = tf.tile(bcol0param,[batch_size,1,1])
        
        A0 = tf.matmul(Acol0param, C_src)
        A1 = tf.matmul(Acol1param, C_src)
        A2 = Acol2
        A3 = tf.matmul(Acol3param, C_src)
        A4 = tf.matmul(Acol4param, C_src)
        A5 = Acol5
        A6 = tf.matmul(Acol61param, C_dst) * tf.matmul(Acol62param, C_src)
        A7 = tf.matmul(Acol71param, C_dst) * tf.matmul(Acol72param, C_src)

        A = tf.transpose(tf.stack([tf.reshape(A0,[-1,8]),tf.reshape(A1,[-1,8]),\
                     tf.reshape(A2,[-1,8]),tf.reshape(A3,[-1,8]),\
                     tf.reshape(A4,[-1,8]),tf.reshape(A5,[-1,8]),\
                     tf.reshape(A6,[-1,8]),tf.reshape(A7,[-1,8])], 1), [0,2,1])
        b = tf.matmul(bcol0param, C_dst)
        
        h8 = tf.matrix_solve(A, b)
        h1 = tf.ones([batch_size, 1, 1])
        H = tf.concat([h8,h1],1)
        H = tf.reshape(H, (-1,3,3))

        
        #H1 = tf.constant([[1., 0., C_a[0]], [0., 1., C_a[1]], [0., 0., 1]], tf.float32)
        #H2 = tf.constant([[1., 0., -C_a[0]], [0., 1., -C_a[1]], [0., 0., 1]], tf.float32)
        #H = tf.matmul(H2, tf.matmul(H, H1))

        #H = tf.matrix_inverse(H)

        #H_fake = tf.eye(3)
        #H_fake = tf.expand_dims(H_fake, [0])
        #H_fake = tf.tile(H_fake,[batch_size,3,3])

        return H
    
    
class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialTransformer, self).__init__()

    def call(self, inputs):
        [image_input,H] = inputs
        image_input = image_input[:,:,:,0:1]
        result = batch_transformer(image_input,H,[128,128])[0]
        result = tf.reshape(result,(-1,128,128,1))
        return result


def getUnsupervisedModel():
    image_input = tf.keras.layers.Input(shape=(128,128,2), name='images')  # Variable-length sequence of ints
    Ca_input = tf.keras.layers.Input(shape=(1,8), name='corners')

    # Homography net
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV1_3x3_64")(image_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV2_3x3_64")(x)
    x = layers.MaxPooling2D((2, 2), name="MP1_2x2")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV3_3x3_64")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="CV4_3x3_64")(x)
    x = layers.MaxPooling2D((2, 2), name="MP2_2x2")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV5_3x3_128")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV6_3x3_128")(x)
    x = layers.MaxPooling2D((2, 2), name="MP3_2x2")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV7_3x3_128")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="CV8_3x3_128")(x)
    
    x = layers.Flatten(name="Flatten")(x)
    x = layers.Dense(1024, activation='relu', name="Dense_1024")(x)
    x = layers.Dense(8, activation='tanh', name="Dense_8")(x)
    
    
    #offset = tf.Variable(initial_value=trainLabel[0])
    #C_a = tf.Variable(initial_value=trainCaList[0])
    H = TensorDLT()(inputs=[x, Ca_input])
    
    '''
    U = tf.Variable(initial_value=trainData[0:2,:,:,:])
    myv = math.sqrt(2)/2
    identity = np.array([[[1., 0., 64.], [0., 1., 64.], [0., 0., 1.]],[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
    identity = identity.flatten()
    identity = identity.reshape([2,1,9])
    theta = tf.Variable(initial_value=identity)
    '''
    #H = layers.Lambda(tf.reshape)(H, [-1,1,9])
    #image_input = tf.reshape(image_input, (128,128,128,2))
    #H = tf.reshape(H, (128,1,9))
    result = SpatialTransformer()(inputs=[image_input,H])
    
    '''
    print(result)
    aa = K.eval(result[0])
    cv2.imshow("in", trainData[0,:,:,0].astype('uint8'))
    cv2.imshow("out", aa[0,:,:,0].astype('uint8'))
    cv2.waitKey(0)


    '''
    
    model = models.Model(inputs=[image_input,Ca_input], outputs=[result,H], name='UnsupervisedModel')
    optimizer = tf.keras.optimizers.Adam(lr = 0.0001, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss=[tf.losses.absolute_difference, tf.losses.mean_squared_error],
                  loss_weights=[1., 0.])

    return model

#*************************************************************************

def outPutW(weights, border=4):
    for w in weights:
        if len(w.shape) == 1:
            print(w[1:border])
        if len(w.shape) == 2:
            print(w[1:border, 1:border])
        if len(w.shape) == 3:
            print(w[1:border, 1:border, 1:border])

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        nothing = 0
        #outPutW(self.model.get_weights())
        #print(tf.keras.backend.get_value(self.model.layers[-2].output))
        step = 10
        if (epoch % step == 0):
            optimizer = tf.keras.optimizers.Adam(lr = (1e-1)/(2**(epoch/step)), amsgrad=False)
            self.model.compile(optimizer=optimizer,
                          loss=[tf.losses.absolute_difference, tf.losses.mean_squared_error],
                          loss_weights=[1., 0.])
        print("lr !!!!!!!!! ", tf.keras.backend.get_value(self.model.optimizer.lr))

def main():
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    np.random.seed(0)

    mode = 1 # 1:train supervised model; 2:train unsupervised model
    trainData, trainLabel, trainCaList, trainHList,_,_ = readImgAndGeneData("../Data/Train/", showImg = False)
    testData, testLabel, testCaList, testHList,_,_ = readImgAndGeneData("../Data/Val/", showImg = False)
    
    # Normalize pixel values to be between 0 and 1
    trainData, testData = trainData / 255.0, testData / 255.0
    trainLabel, testLabel = trainLabel / float(MAX_PERTURBATION), testLabel / float(MAX_PERTURBATION)

    
    '''
    offset = tf.Variable(initial_value=trainLabel)
    C_a = tf.Variable(initial_value=trainCaList)
    H = TensorDLT()(inputs=[offset, C_a])
    for i in range(len(trainHList)):
        print(np.max(np.abs(K.eval(H)[i]-trainHList[i])))
        print(K.eval(H)[i]-trainHList[i])
    print(np.max(np.square((K.eval(H)-trainHList))))
    
    sin45 = math.sqrt(2) / 2
    #identity = np.array([[1., 0., 60.], [0., 1., 60.], [0., 0., 1.]])
    #identity = np.array([[sin45, -sin45, 0.], [sin45, sin45, 0.], [0., 0., 1.]])
    identity = np.array([[sin45, -sin45, 20.], [sin45, sin45, 20.], [0., 0., 1.]])
    #identity = np.linalg.inv(identity)
    #warpimg = cv2.warpPerspective(trainData[0,:,:,0], np.linalg.inv(identity), (128,128))
    #cv2.imshow("warpimg", warpimg)

    identity = identity.flatten()
    theta = tf.Variable(initial_value=identity)
        
    #result = SpatialTransformer()(inputs=[trainData[0:1],theta])
    
    warpimgA = cv2.warpPerspective(trainData[0,:,:,0], np.linalg.inv(trainHList[0]), (128,128))
    warpimgB = cv2.warpPerspective(trainData[0,:,:,1], trainHList[0], (128,128))
    cv2.imshow("cv warpimgA", warpimgA)
    cv2.imshow("cv warpimgB", warpimgB)
    #trainData[:,:,:,0] = trainData[:,:,:,1]
    result = SpatialTransformer()(inputs=[trainData,trainHList])
    cv2.imshow("Patch A", trainData[0,:,:,0])
    cv2.imshow("network Warped Patch A", K.eval(result[0,:,:,0]))
    cv2.imshow("Patch B", trainData[0,:,:,1])
    cv2.waitKey(0)
    '''

    if mode == 1:
        model = getSupervisedModel()  # for supervised training
    else:
        model = getUnsupervisedModel()
    
    
    model.summary()

    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    #cwd = os.getcwd()
    filepath = "checkpoints\\cp-{epoch:02d}.hdf5"
    #filepath = "checkpoints\\checkpoint.hdf5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
    
    if mode == 1:
    # for supervised training
        history = model.fit(trainData, trainLabel, epochs=100, batch_size=BATCH_SIZE,
                            validation_data=(testData, testLabel),
                            callbacks=[tensorboard_callback, checkpoint_callback]) #, MyCustomCallback()
    else:
        history = model.fit({'images':trainData, 'corners':trainCaList}, 
                            {'spatial_transformer':trainData[:, :, :, 1:2], 'tensor_dlt':trainHList}, epochs=200, batch_size=128,
                            validation_data=({'images':testData, 'corners':testCaList}, 
                            {'spatial_transformer':testData[:, :, :, 1:2], 'tensor_dlt':testHList}),
                            callbacks=[tensorboard_callback, checkpoint_callback, MyCustomCallback()]) #


    '''
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    '''
    
    #predictedData = model.predict({'images':testData, 'corners':testCaList})
    #test_loss = model.evaluate({'images':testData, 'corners':testCaList}, 
    #                           {'spatial_transformer':testData[:, :, :, 1:2], 'tensor_dlt':testHList}, verbose=2)

    #print(test_loss)


if __name__ == '__main__':
    main()
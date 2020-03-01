from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from Train import readImgAndGeneData, TensorDLT, SpatialTransformer
from tensorflow.losses import absolute_difference, mean_squared_error
import tensorflow.losses
#from Wrapper import ANMS, FeatureList, FeatureMatching, EstimateHomography, DrawFeatureMatching
import glob
import os



def showPredictRes(predictedData, testLabel, imgList, rectList, outPath='', maxOffset=32, numPairsInOneImg=8, getEPEonly=False):
    EPE = 0
    for idx in range(len(testLabel)):
        ymin = rectList[idx][0]
        ymax = rectList[idx][1]-1
        xmin = rectList[idx][2]
        xmax = rectList[idx][3]-1
        C_a = [(xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ]
        C_b_predicted = [(int(xmin+predictedData[idx][0]*maxOffset), int(ymin+predictedData[idx][1]*maxOffset)),
                (int(xmax+predictedData[idx][2]*maxOffset), int(ymin+predictedData[idx][3]*maxOffset)),
                (int(xmax+predictedData[idx][4]*maxOffset), int(ymax+predictedData[idx][5]*maxOffset)),
                (int(xmin+predictedData[idx][6]*maxOffset), int(ymax+predictedData[idx][7]*maxOffset))
            ]
        C_b_groundtruth = [(int(xmin+testLabel[idx][0]*maxOffset), int(ymin+testLabel[idx][1]*maxOffset)),
                (int(xmax+testLabel[idx][2]*maxOffset), int(ymin+testLabel[idx][3]*maxOffset)),
                (int(xmax+testLabel[idx][4]*maxOffset), int(ymax+testLabel[idx][5]*maxOffset)),
                (int(xmin+testLabel[idx][6]*maxOffset), int(ymax+testLabel[idx][7]*maxOffset))
            ]
        
        for i in range(4):
            EPE += np.linalg.norm(np.array(C_b_groundtruth[i]) - np.array(C_b_predicted[i]))

        if getEPEonly == False:
            img = imgList[int(idx / numPairsInOneImg)].copy()
            for i in range(4):
                img = cv2.line(img, C_a[i], C_a[(i+1)%4], (255, 0, 0), 2)
                img = cv2.line(img, C_b_predicted[i], C_b_predicted[(i+1)%4], (0, 0, 255), 4)
                img = cv2.line(img, C_b_groundtruth[i], C_b_groundtruth[(i+1)%4], (0, 255, 0), 2)
            
            if idx % numPairsInOneImg == 0:
                fileName = outPath + str(int(idx / numPairsInOneImg)) + '_' + str(idx % numPairsInOneImg) + '.jpg'
                cv2.imwrite(fileName, img)
                cv2.imshow("img", img)
                #cv2.waitKey(0)
                
    return EPE / len(testLabel) / 4


def getEPE(predictedData, testLabel, rectList, maxOffset=32, numPairsInOneImg=8):
    EPE = 0
    for idx in range(len(predictedData)):
        ymin = rectList[idx][0]
        ymax = rectList[idx][1]-1
        xmin = rectList[idx][2]
        xmax = rectList[idx][3]-1
        C_b_predicted = np.array([(int(xmin+predictedData[idx][0]*maxOffset), int(ymin+predictedData[idx][1]*maxOffset)),
                (int(xmax+predictedData[idx][2]*maxOffset), int(ymin+predictedData[idx][3]*maxOffset)),
                (int(xmax+predictedData[idx][4]*maxOffset), int(ymax+predictedData[idx][5]*maxOffset)),
                (int(xmin+predictedData[idx][6]*maxOffset), int(ymax+predictedData[idx][7]*maxOffset))
            ])
        C_b_groundtruth = np.array([(int(xmin+testLabel[idx][0]*maxOffset), int(ymin+testLabel[idx][1]*maxOffset)),
                (int(xmax+testLabel[idx][2]*maxOffset), int(ymin+testLabel[idx][3]*maxOffset)),
                (int(xmax+testLabel[idx][4]*maxOffset), int(ymax+testLabel[idx][5]*maxOffset)),
                (int(xmin+testLabel[idx][6]*maxOffset), int(ymax+testLabel[idx][7]*maxOffset))
            ])
        EPE += np.linalg.norm(C_b_groundtruth - C_b_predicted)

    return EPE / len(predictedData) / 4



def showPredictResWithH(predictedData, testLabel, testHList, imgList, rectList, outPath='', maxOffset=32, numPairsInOneImg=8, getEPEonly=False):
    EPE = 0
    for idx in range(len(testLabel)):
        ymin = rectList[idx][0]
        ymax = rectList[idx][1]-1
        xmin = rectList[idx][2]
        xmax = rectList[idx][3]-1
        C_a = [(xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax)
            ]
        
        C_a_m = [(0, 0, 1),
                (127, 0, 1),
                (127, 127, 1),
                (0, 127, 1)
            ]
        C_a_m = np.transpose(C_a_m)
        H = predictedData[idx]
        C_b = np.matmul(H, C_a_m)
        #C_b = np.matmul(testHList[idx], C_a_m)
        C_b_predicted = [(int(C_b[0,0]/C_b[2,0] + xmin), int(C_b[1,0]/C_b[2,0] + ymin)),
                (int(C_b[0,1]/C_b[2,1] + xmin), int(C_b[1,1]/C_b[2,1] + ymin)),
                (int(C_b[0,2]/C_b[2,2] + xmin), int(C_b[1,2]/C_b[2,2] + ymin)),
                (int(C_b[0,3]/C_b[2,3] + xmin), int(C_b[1,3]/C_b[2,3] + ymin))
            ]
        C_b_groundtruth = [(int(xmin+testLabel[idx][0]*maxOffset), int(ymin+testLabel[idx][1]*maxOffset)),
                (int(xmax+testLabel[idx][2]*maxOffset), int(ymin+testLabel[idx][3]*maxOffset)),
                (int(xmax+testLabel[idx][4]*maxOffset), int(ymax+testLabel[idx][5]*maxOffset)),
                (int(xmin+testLabel[idx][6]*maxOffset), int(ymax+testLabel[idx][7]*maxOffset))
            ]
        for i in range(4):
            EPE += np.linalg.norm(np.array(C_b_groundtruth[i]) - np.array(C_b_predicted[i]))

        if getEPEonly == False:
            img = imgList[int(idx / numPairsInOneImg)].copy()
            for i in range(4):
                img = cv2.line(img, C_a[i], C_a[(i+1)%4], (255, 0, 0), 2)
                img = cv2.line(img, C_b_predicted[i], C_b_predicted[(i+1)%4], (0, 0, 255), 4)
                img = cv2.line(img, C_b_groundtruth[i], C_b_groundtruth[(i+1)%4], (0, 255, 0), 2)
            
            #cv2.imshow("img", img)
            #cv2.waitKey(0)

            if idx % numPairsInOneImg == 0:
                fileName = outPath + str(int(idx / numPairsInOneImg)) + '_' + str(idx % numPairsInOneImg) + '.jpg'
                cv2.imwrite(fileName, img)
                #cv2.imshow("img", img)
                #cv2.waitKey(0)

    return EPE / len(testLabel) / 4

'''
def getClassicalH(imagePairs):
    resize_image_size = 500
    size_of_image = (resize_image_size,resize_image_size)
    Translation = (300,300)
    ANMS_corners = 180
    H_list = []
    for imagePair in imagePairs:
        img1 = imagePair[:,:,0].astype('uint8')
        img2 = imagePair[:,:,1].astype('uint8')
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey(1)

        Corners1,Corners2 = Corners(img1, img2)
        #Corners1 = ANMS(img1,10)
        #Corners2 = ANMS(img2,8)
        Feature1 = FeatureList(img1,Corners1)
        Feature2 = FeatureList(img2,Corners2)

        kp1_2 , kp2 = FeatureMatching(Feature1,Feature2,1.5)

        H,fpt1,fpt2 = EstimateHomography(kp1_2,kp2,0.5)
        H_list.append(H)
        #DrawFeatureMatching(img1,img2,fpt1,fpt2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return H_list
'''

def getHFromH4(predictedData):
    ymin = 0
    ymax = 127
    xmin = 0
    xmax = 127
    C_a = [(xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax)
        ]
    idx = 0
    maxOffset = 32
    C_b_predicted = [(int(xmin+predictedData[idx][0]*maxOffset), int(ymin+predictedData[idx][1]*maxOffset)),
            (int(xmax+predictedData[idx][2]*maxOffset), int(ymin+predictedData[idx][3]*maxOffset)),
            (int(xmax+predictedData[idx][4]*maxOffset), int(ymax+predictedData[idx][5]*maxOffset)),
            (int(xmin+predictedData[idx][6]*maxOffset), int(ymax+predictedData[idx][7]*maxOffset))
        ]
    
    H = cv2.getPerspectiveTransform(np.array(C_b_predicted, dtype = "float32"), np.array(C_a, dtype = "float32"))
    return H

def getHsupervised(model, img1_ori_grey, img2_ori_grey): # image shape can be (w, h, 1) or (w, h)
    image_size = 128
    img1 = cv2.resize(img1_ori_grey,(image_size,image_size)) / 255.0
    img2 = cv2.resize(img2_ori_grey,(image_size,image_size)) / 255.0
    img_pair = np.stack([img1,img2], axis=2)
    img_pair = np.reshape(img_pair, (1, image_size, image_size, 2))
    
    predictedData = model.predict(img_pair)
    H = getHFromH4(predictedData)
    
    H1 = [[img1_ori_grey.shape[0] / img1.shape[0], 0, 0],[0, img1_ori_grey.shape[1] / img1.shape[1], 0], [0,0,1]]
    H2 = [[img2_ori_grey.shape[0] / img2.shape[0], 0, 0],[0, img2_ori_grey.shape[1] / img2.shape[1], 0], [0,0,1]]
    H = np.matmul(H2, np.matmul(H, np.linalg.inv(H1)))
    return H

def getHunsupervised(model, img1_ori_grey, img2_ori_grey): # image shape can be (w, h, 1) or (w, h)
    image_size = 128
    img1 = cv2.resize(img1_ori_grey,(image_size,image_size)) / 255.0
    img2 = cv2.resize(img2_ori_grey,(image_size,image_size)) / 255.0
    img_pair = np.stack([img1,img2], axis=2)
    img_pair = np.reshape(img_pair, (1, image_size, image_size, 2))
    corners = np.zeros((1,1,8))
    
    predictedData = model.predict({'images':img_pair, 'corners':corners})
    H = predictedData[1][0]
    
    '''
    insideWarpImage = predictedData[0][0]
    cv2.imshow('insideWarpImage', insideWarpImage)
    warpImg1 = cv2.warpPerspective(img1, np.linalg.inv(H), (img2.shape[1], img2.shape[0]))
    cv2.imshow('warpImg1', warpImg1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    '''
    
    H1 = [[img1_ori_grey.shape[0] / img1.shape[0], 0, 0],[0, img1_ori_grey.shape[1] / img1.shape[1], 0], [0,0,1]]
    H2 = [[img2_ori_grey.shape[0] / img2.shape[0], 0, 0],[0, img2_ori_grey.shape[1] / img2.shape[1], 0], [0,0,1]]
    H = np.matmul(H2, np.matmul(np.linalg.inv(H), np.linalg.inv(H1)))
    return H


def sample():
    folderPath = '../../P1TestSet/Phase1/TestSet1'
    fileNames = glob.glob(folderPath + "/*.jpg")

    
    # invoke getHsupervised like this
    filepath = ".\\checkpoints\\Supervised_Model.hdf5"
    model = tf.keras.models.load_model(filepath)
    
    img1_ori_grey = cv2.imread(fileNames[0], 0)
    img2_ori_grey = cv2.imread(fileNames[1], 0)

    H = getHsupervised(model, img1_ori_grey, img2_ori_grey) # API FUNCTION
    
    warpImg_ori1 = cv2.warpPerspective(img1_ori_grey, H, (img2_ori_grey.shape[1], img2_ori_grey.shape[0]))
    cv2.imshow('supv_warpImg_ori1', warpImg_ori1)
    cv2.imshow('supv_img1_ori_grey', img1_ori_grey)
    cv2.imshow('supv_img2_ori_grey', img2_ori_grey)
    cv2.waitKey(0)
    


    
    # invoke getHunsupervised like this
    filepath = ".\\checkpoints\\Unsupervised_Model.hdf5"
    model = tf.keras.models.load_model(filepath, custom_objects={'TensorDLT': TensorDLT, 'SpatialTransformer': SpatialTransformer, 'absolute_difference':absolute_difference})
    
    
    img1_ori_grey = cv2.imread(fileNames[0], 0)
    img2_ori_grey = cv2.imread(fileNames[1], 0)

    H = getHunsupervised(model, img1_ori_grey, img2_ori_grey) # API FUNCTION
    
    warpImg_ori1 = cv2.warpPerspective(img1_ori_grey, H, (img2_ori_grey.shape[1], img2_ori_grey.shape[0]))
    cv2.imshow('unsupv_warpImg_ori1', warpImg_ori1)
    cv2.imshow('unsupv_img1_ori_grey', img1_ori_grey)
    cv2.imshow('unsupv_img2_ori_grey', img2_ori_grey)
    cv2.waitKey(0)

'''
def stitch():
    filepath = ".\\checkpoints\\Supervised_Model.hdf5"
    model = tf.keras.models.load_model(filepath)

    for testNum in range(4):
        path = '../../P1TestSet/Phase1/TestSet'+str(testNum+1)+'/'
        print(path)
        images=[]
        for img in os.listdir(path):
            images.append(img)
        images.sort()
    
        warped_images=[]

        size_of_image = (1000,1000)
        resize_image_size = 500
        Translation = (100,100)
        ANMS_corners = 180
    
        ref_img = cv2.imread("%s%s" % (path, images[0]))
        ref_img = cv2.resize(ref_img, (resize_image_size,resize_image_size), interpolation = cv2.INTER_AREA)
        ref_img_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
        #ref_corner = ANMS(ref_img_gray , ANMS_corners)
        #ref_Feature = FeatureList(ref_img_gray,ref_corner)
        ref_H = np.matmul(np.array([[1,0,Translation[0]],[0,1,Translation[1]],[0,0,1]]),np.identity(3))
        warped_images.append(cv2.warpPerspective(ref_img, ref_H, size_of_image))
    
        for i in range(1,len(images)):
            print(i)
            img = cv2.imread("%s%s" % (path, images[i]))
            img = cv2.resize(img, (resize_image_size,resize_image_size), interpolation = cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #corners = ANMS(img_gray,ANMS_corners)
            #features = FeatureList(img_gray,corners)
            #kp1 , kp2 = FeatureMatching(ref_Feature,features,0.8)
            #H , matched_pts1, matched_pts2 = EstimateHomography(kp1 , kp2,2)

            H = getHsupervised(model, ref_img_gray, img_gray) # API FUNCTION

            ref_H = np.matmul(ref_H , np.linalg.inv(H))
            warped_images.append(cv2.warpPerspective(img, ref_H, size_of_image))
            #ref_Feature = features
        
        out =  np.zeros((size_of_image[1],size_of_image[0],3),dtype=np.uint8)
        temp = np.array([0,0,0],dtype=np.uint8)
        out = np.copy(warped_images[0])
        for img in range(1,len(warped_images)):
            for i in range(size_of_image[1]):
                for j in range(size_of_image[0]):
                    if (np.array_equal(warped_images[img][i][j],temp) == False):
                        out[i][j] = warped_images[img][i][j]
    
        cv2.imwrite(path + 'Output_supv.jpg' , out)
        cv2.imshow('Output' , out)
        cv2.destroyAllWindows()
'''

def main():
    #stitch()
    #return 0
    np.random.seed(0)
    dataTypes = ['Train', 'Val', 'Test']
    for dataType in dataTypes:
        #trainData, trainLabel = readImgAndGeneData("../Data/Train/", showImg = False)
        testData, testLabel, testCaList, testHList, imgList, rectList = readImgAndGeneData("../Data/"+dataType+"/", showImg = False)
        #testData, testLabel, imgList, rectList = readImgAndGeneData("../Data/Val0/", showImg = False)

        testData, testLabel = testData / 255.0, testLabel / 32.0
    
        filepath = ".\\checkpoints\\Supervised_Model.hdf5"
        model = tf.keras.models.load_model(filepath)
        #model.summary()
    
        predictedData = model.predict(testData)
        outPath = '../Data/'+dataType+'_res_sup/'
        EPE = showPredictRes(predictedData, testLabel, imgList, rectList, outPath, getEPEonly=False)
        print('Supervised_Model', dataType, 'EPE=', EPE)
    

        filepath = ".\\checkpoints\\Unsupervised_Model.hdf5"
        model = tf.keras.models.load_model(filepath, custom_objects={'TensorDLT': TensorDLT, 'SpatialTransformer': SpatialTransformer, 'absolute_difference':absolute_difference})
        #model.summary()

        predictedData = model.predict({'images':testData, 'corners':testCaList})
        outPath = '../Data/'+dataType+'_res_unsup/'
        EPE = showPredictResWithH(predictedData[1], testLabel, testHList, imgList, rectList, outPath, getEPEonly=False)
        print('Unsupervised_Model', dataType, ' EPE=', EPE)
        


if __name__ == '__main__':
    main()
#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1

"""

# Code starts here:

import numpy as np
import sys
import os
import cv2
# Add any python libraries here


def ANMS(img,best_corners):
	corners = cv2.goodFeaturesToTrack(img,10000,0.01,10)
	print corners.shape
	if (corners.shape[0]<=best_corners):
		print("ANMS corners are more than corners found in the image. \n Please decrease ANMS_corners value on line 156 and Try Again ")
		sys.exit()
	total_corners = corners.shape[0]
	C_score = cv2.cornerHarris(img,2,3,0.04)
	r = np.inf * np.ones((total_corners,3))
	r_i = np.inf
	ED = 0
	for i in range(total_corners):
		for j in range(total_corners):
			x_i = int(corners[i][0][0])
			y_i = int(corners[i][0][1])
			x_j = int(corners[j][0][0])
			y_j = int(corners[j][0][1])
			if (C_score[y_i][x_i] > C_score[y_j][x_j]):
				ED = (x_j - x_i)**2 + (y_j - y_i)**2
			if ED < r[i][2]:
				r[i][0] = x_i
				r[i][1] = y_i
				r[i][2] = ED
	r = r[r[:,2].argsort()]
	r = r.astype(int)
	output_corners = np.zeros((best_corners,2))
	j=0
	i=0
	flag = True
	print r.shape
	while(j<best_corners):
		try:
			if (r[-i][0] > 21 and r[-i][0]< img.shape[1]-21 and r[-i][1]>21 and r[-i][1]< img.shape[0]-21) :
				output_corners[j] = np.array([r[-i][0] , r[-i][1]])
				j=j+1
			i=i+1
		except:
			flag = False
			break
	if flag==False:
		print("ANMS corners are more than corners found in the image. \n Please decrease ANMS_corners value on line 156 and Try Again ")
		sys.exit()
	return output_corners.astype(int)

def FeatureList(img,corners):
	features = {}
	
	for i in range(corners.shape[0]):
		area = img[corners[i][1]-20:corners[i][1]+20, corners[i][0]-20:corners[i][0]+20]
		area = cv2.GaussianBlur(area,(5,5),cv2.BORDER_DEFAULT)
		area_vec = cv2.resize(area,(8,8))
		feature_vec = np.resize(area_vec , (64,1))
		features[(corners[i][0],corners[i][1])] = (feature_vec-np.mean(feature_vec)) / np.std(feature_vec)
	return features

def FeatureMatching(features1,features2,threshold):
	flag =True
	kp1 = []
	kp2 = []
	for feature_point1 in features1:
		dist_list = []
		for feature_point2 in features2:
			dist_list.append((feature_point2 , np.linalg.norm(features1[feature_point1]-features2[feature_point2])))
		dist_list.sort(key = lambda X:X[1])
		if ( (dist_list[0][1] / dist_list[1][1]) < threshold):
			kp1.append(feature_point1)
			kp2.append(dist_list[0][0])
	if len(kp1)<=4 or len(kp2)<=4:
		flag = False
	return kp1,kp2,flag

def DrawFeatureMatching(strng,img1,img2,kp1,kp2):  # img1 and img2 must be Color images 
	new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
	new_img = np.zeros(new_shape, type(img1.flat[0]))
	# Place images onto the new image.
	new_img[0:img1.shape[0],0:img1.shape[1]] = img1
	new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
	for i in range(len(kp1)):
		x1 = kp1[i][0]
		y1 = kp1[i][1]
		x2 = kp2[i][0] + int(img1.shape[1])
		y2 = kp2[i][1]
		cv2.line(new_img,(x1,y1),(x2,y2),(0,0,255),1)
		strng = '%s%s' % (strng,'.jpg')
	cv2.imwrite(strng,new_img)

def EstimateHomography(kp1 , kp2 , threshold):
	flag = True
	maxinlier = 0
	for itr in range(0,8000):
		index = np.random.randint(len(kp1),size=4)
		pts1 = np.zeros((4,2),dtype=np.float32)
		pts2 = np.zeros((4,2),dtype=np.float32)
		for i in range(len(index)):
			pts1[i][1] = kp1[index[i]][1]
			pts1[i][0] = kp1[index[i]][0]
			pts2[i][1] = kp2[index[i]][1]
			pts2[i][0] = kp2[index[i]][0]
		H,status = cv2.findHomography(pts1,pts2)
		inlier = 0
		matched_pts1 = []
		matched_pts2 = []
		for i in range(len(kp1)):
			source_pts = np.array([kp1[i][0],kp1[i][1],1])
			actual_pts = np.array([kp2[i][0],kp2[i][1],1])
			try:
				predicted_pts = np.matmul(H, source_pts)
			except:
				break
			predicted_pts = predicted_pts / predicted_pts[2]

			if(np.linalg.norm(actual_pts - predicted_pts) < threshold):
				matched_pts1.append(source_pts)
				matched_pts2.append(actual_pts)
				inlier = inlier + 1
		if(inlier > maxinlier):
			maxinlier = inlier
			final_H = H
			final_pts1 = matched_pts1
			final_pts2 = matched_pts2

	print("Max Inlier = ",maxinlier)
	if (maxinlier<=4):
		print("Total Inliers are less than 4")
		flag = False
	f_pts1 = np.zeros((maxinlier,2),dtype=np.float32)
	f_pts2 = np.zeros((maxinlier,2),dtype=np.float32)
	for i in range(maxinlier):
		f_pts1[i][0] = final_pts1[i][0]
		f_pts1[i][1] = final_pts1[i][1]
		f_pts2[i][0] = final_pts2[i][0]
		f_pts2[i][1] = final_pts2[i][1]
	final_H, status = cv2.findHomography(f_pts1,f_pts2)
	return final_H,final_pts1,final_pts2,flag

def main():
	images=[]
	path = "/home/raghav/Desktop/YourDirectoryID_p1/P1TestSet/TestSet2Arranged/"
	for img in os.listdir(path):
		images.append(img)
	images.sort()
	print images
	warped_images=[]

	size_of_image = (6000,1500)
	resize_image_size = 500
	Translation = (100,150)
	ANMS_corners = 150

	ref_img = cv2.imread("%s%s" % (path, images[0]))
	print(ref_img.shape)
	ref_img = cv2.resize(ref_img, (resize_image_size,resize_image_size), interpolation = cv2.INTER_AREA)
	ref_img_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)
	ref_corner = ANMS(ref_img_gray , ANMS_corners)
	ref_Feature = FeatureList(ref_img_gray,ref_corner)
	ref_H = np.matmul(np.array([[1,0,Translation[0]],[0,1,Translation[1]],[0,0,1]]),np.identity(3))
	warped_images.append(cv2.warpPerspective(ref_img, ref_H, size_of_image))

	for i in range(1,len(images)):
		print "For  ",i,".jpg and ",i+1,".jpg image"
		img = cv2.imread("%s%s" % (path, images[i]))
		img = cv2.resize(img, (resize_image_size,resize_image_size), interpolation = cv2.INTER_AREA)
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		corners = ANMS(img_gray,ANMS_corners)
		features = FeatureList(img_gray,corners)
		kp1 , kp2,flag = FeatureMatching(ref_Feature,features,0.8)
		if flag == False:
			print "Not enough correspondencies between ",i,"th and ",i+1,"th image \n Generating output till",i,"th image ."
			break
		DrawFeatureMatching("Feature_matching",ref_img,img, kp1,kp2)
		H , matched_pts1, matched_pts2,flag = EstimateHomography(kp1 , kp2,2)
		if flag == False:
			print "Not able to compute Homography between ",i,"th and ",i+1,"th image \n Generating output till",i,"th image ."
			break
		DrawFeatureMatching("RANSAC",ref_img,img,matched_pts1, matched_pts2)
		ref_H = np.matmul(ref_H , np.linalg.inv(H))
		warped_images.append(cv2.warpPerspective(img, ref_H, size_of_image))
		ref_img = np.copy(img)

		ref_Feature = features
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		
	out =  np.zeros((size_of_image[1],size_of_image[0],3),dtype=np.uint8)
	temp = np.array([0,0,0],dtype=np.uint8)
	out = np.copy(warped_images[0])
	for img in range(1,len(warped_images)):
		for i in range(size_of_image[1]):
			for j in range(size_of_image[0]):
				if (np.array_equal(warped_images[img][i][j],temp) == False):
					out[i][j] = warped_images[img][i][j]
	
	# cv2.imwrite('Output.jpg' , out)
	cv2.imwrite('Output_Train_Set3.jpg' , out)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
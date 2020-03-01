Project 1: My Autopano

Raghav Nandwani

Phase 1: 
To appropriately run the code you need to make following changes.

Insert the correct path of directory at line 156

Make these changes at line 163-166 for appropriate output. or all outputs are generated using the following parameters. 

TRAIN SET : 
Set 1 
	size_of_image = (900,1000) i.e. Canvas size
	resize_image_size = 500 i.e. rescaling size of the image ( all the images will be converted into a square of size (resize_image_size X resize_image_size)
	Translation = (300,100) i.e. Translation vector with which the first image will shift as all the images will be projected on this image only
	ANMS_corners = 400 i.e. Number of corners after ANMS extraction

Set 2 
	size_of_image = (2000,1000)
	resize_image_size = 500
	Translation = (300,300)
	ANMS_corners = 400

Set 3 
	size_of_image = (3000,1500)
	resize_image_size = 500
	Translation = (300,100)
	ANMS_corners = 180

TEST SET :

For all sets
	size_of_image = (3000,1500)
	resize_image_size = 500
	Translation = (300,100)
	ANMS_corners = 150

For all sets the threshold for the ratio of Norm for the feature matching and the Homography estimation is kept constant at 0.8 and 2 respectively. If you want to change it, it can be done at line 184 and 189 respectively. (Note : it will be same for all the images in one directory)

The program will generate following output
1. Feature Correspondencies in the images
2. Feature Correspondencies in the images after RANSAC
3. Output of the stitched panorama

I have also included the TestSet2Arranged folder, that contains the TestSet2 images in the correct order so that after every subsequent image we have some correspondences.

Phase 2
For Train.py, set the training model in line 377 (1:train supervised model; 2:train unsupervised model), and set the path of Train and Val folder in line 378, 379.
For Test.py, just set the data path in line 331. It will run supervised model and unsupervised model over the train, val, and test folder.

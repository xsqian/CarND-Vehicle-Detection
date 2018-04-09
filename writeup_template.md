

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample_car_noncar_images.png
[image2]: ./output_images/visual_of_hog.png
[image3]: ./output_images/test_of_find_cars.png
[image4]: ./output_images/test_with_different_scan_zones.png
[image5]: ./output_images/headmap_and_detected_cars.png
[image6]: ./output_images/different_parameters.png
[video1]: ./project_video_out.mp4


### Project Report

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The scikit-image package has a built in function to extract Histogram of Oriented Gradient features from an image. The code for this step is contained in the 3rd code cell of the IPython notebook in the get_hog_features() function. In the get_hog_features() function, we call hog() function from the skiamge.feature package. The parameters pass to the hog function are orientations, pixels_per_cell, cells_per_block, transform_sqrt, visualise and feture_vector.

I started by reading in all the `vehicle` and `non-vehicle` images from the large dataset.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different parameters for `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I used the extract_features() function with different color space and tried various combinations of parameters to extract the hog features and training the classifier. With many different combinations, I found that colorspace "LUV" and hog_channel "ALL" had the highest test accuracy.

![alt text][image6]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I first used used "LUV" and "ALL" as the classifier(model) to train the classifier and detect the cars in the video, I noticed that there a a lot of false positives. After many many experiment, I found the "HSV" colorspace and "ALL" hog_channel achieved better result.
I trained a linear SVC using color_space="HSV", hog_channel="ALL", orientations=9, pixels_per_cell=16, cells_per_block=2. The extracted features has a size of 1188. The total number of images for the training and testing are 17,760, 8792 of car images and 8968 of noncar images. I split the training dataset to be 80% and 20% of dataset is used for testing. The code for the SVC training is in the 8th cell of the IPython notebook.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions start around 400 pixel in y direction since the bottom section of the image is more related to the surrounding of the driving car. The top half of the image is far in the horizon or just the sky. In the middle of the image, I used the smaller scales for the sliding windows since the car image would be smaller because they are more far away. As we move down closer to the bottom of the image, the sliding windows are bigger since cars are much closer.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some of the examples that I tested the classifier on the testing images. I spent quite sometime on combining hog features with bin_spatial and color_hist features for the classifier, however, many many experiments didn't show me much improvement on classier testing accuracy. So I simplified the classifier to HOG features only, but with "ALL" color channels.
Ultimately I searched on 5 different scales (1, 1.5, 2, 3.5) using HSV 3-channel HOG features, which provided a nice result.

Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Here's a [link to my video result](./project_video_out.mp4)


#### 2. Filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the original image:

### Here are figures show the added heat, after remove threshold, the heatmap and the detected cars.

![alt text][image5]

Since we are detecting cars in the video, in which the car position from previous frame to the next frame should be in the similar position.
I created a Vehcile() object to keep the car positions detected from the previous frame(s) depending on the length of history parameter we set. For each frame, I first use the classifier to detect the new car positions. If the car position is detected, I will add the new detected position to the Vehicle object. Then I add the current detected positions with historical positions to the heat. Then apply the threshold to the hedmap and scipy.ndimage.measurements label() function.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I first started with the hog features and detected the cars. However, even though the classifier testing accuracy can be as high as 98% accurate, but once I apply the pipeline to the videos, there are a lot more false positive detections. With a high threshold, however, some valid detections would be filtered out also. So I decided to try out the combination of hog_features, bin_spatial and color_hist. With many hours of experiment, I couldn't achieve much improvement on the detection on the video even the classifier test accuracy is in the 98% range. I think this might have something to do with the training dataset vs the project video. If we had some car and non car images from the video that participate in the training, the detection of the cars in the video might be much more accurate.
So I revert back to hog features for the training and detection. The classifier is much faster to training and the feature extraction from the the video is much faster, too.

The pipeline without remembering the previous frame car position is much jittering and unstable, with the implementation of Vehicle object to keep track of the previous detected position, it improved the stability of the pipeline.

The other area of improvement if have more time is to try use multiple models for the same frame on the video and combine the detections from the multiple models. The other solution to compare is to use a Deep Neural Network for the vehicle detection.

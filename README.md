**Image Correction: Planar Motion**

Author: Baigalmaa Bayarsaikhan (MSc Computer Science) 

**1. Task description**

Planar motion is the case when a camera mounted on car, moving on planar road. Producing a result contains image plane perpendicular to the ground would be beneficial. Implement and develop an algorithm that estimates homography, corrects image orientation to be perpendicular. Concept of planar motion is one of the crucial topic in computer vision and autonomous vehicles. Detecting motion vector from moving image frames and analysing rigid or non-rigid objects can diverge different solutions of image processing. In case of, mounted camera is moving through scenes which surrounded by dynamic or static objects. For example, autonomous vehicle situation, car, equipped with camera, can move in street which has buildings, human and other cars. Keep tracking the planar motion in image frames can remain the car under control on the road and correction of planar surface always be required due to fixation of camera perpendicular to ground is not always possible.

**2. Planar Motion**

In computer vision, planar is the surface which represents flat base in the image or road. Planar surface can belong to all the images or frames that have ground
like surfaces. Most of the images in vehicle environment relates with motion of the vehicle on the road which makes planar motion detection crucial in automation field.
In terms of autonomous vehicle, it assume that camera mounted on the moving car and horizontal movement is parallel with the planar surface. To find planar
motion from consecutive image frames, this laboratory work implements motion vector generation with optical flow technique, then producing mathematical model
on motion vectors iteratively using RANSAC algorithm. Eventually, implementation produces planar motion flowing point which estimates planar surface within image
frames.

**3. RANSAC algorithm**

RANSAC is abbreviation of Random Sample Consensus, generates mathematical model on data set with given threshold. The algorithm separates data set to inliers
and outliers depending on model fitting. Inliers are the data where model fits well. On the other hand, Outliers are the ones do not fit in model. Below are the steps
of the algorithm:
	• Select random subset of original data
	• Estimate model on each data set
	• Differentiate inliers and ouliers on fitting model. Model has fitted well with the data, selected then, data will count as inlier, otherwise suppose the data is outlier
	• Iterating over random subset selection and model fitting. Each iteration, number of inliers should be maximal to get best fitted data set. At the end, generated inliers 	produce best data set, fitted with the model.
 
**4. Implementation**

Finding planar motion problem has separated into two parts. One for generating motion vectors for image sequence and the other one is producing planar surface flowing point. Generating motion vectors on each pixel of the image has implemented by OpenCV FarneBack technique. It gets two images as an input and generates altered magnitude and angle for each pixels with dense optical flow calculation. The technique produces more accurate result when image frames are slowly changing, same as implementation assumption. Moreover, more than two frames used in order to get longer length motion vectors, reduce loss of calculation in RANSAC model fitting. Then, RANSAC algorithm aims to get planar model on generated motion vectors between image frames. To run RANSAC algorithm:
	• Random two vectors have selected and calculated intersection point between
	them.
	• Calculated distance between intersection point for all motion vectors
	• Find inliers on the distance measurement with given threshold
	• Choose best inliers and the intersection point along with iteration

In the end of the RANSAC iteration, best inliers and the intersection point from random selection estimates planar motion point in the consecutive image frames. In addition, Longer motion vectors increase the chance of finding distance on each motion vector.

**5. Conclusion**

Estimating a result point close to optimal, RANSAC algorithm iterates over random two motion vector’s intersecting points in each iteration. Rigid objects in the scene direct to same direction and planar surface is an rigid object where assumption can hold for larger number of parallel motion direction vectors lead to best inliers and planar surface. As a result of the implementation, the method approximates planar motion point with more than two image frames.

**6. Result**

According to RANSAC agreements on data set. Inliers depict as green arrowed vectors lines and outliers in red. Yellow dot is the intersection point, generated with random motion vector selection to best inliers, also, it interprets planar motion point.

![image](https://github.com/user-attachments/assets/00503908-d2db-4237-b522-b2fb6a124406)
Figure 1: 20 frames motion from Malaga urban dataset extract 04

![image](https://github.com/user-attachments/assets/cf9d0593-603f-48ea-985e-ee62789f8e34)
Figure 2: 30 frames motion from Malaga urban dataset extract 01

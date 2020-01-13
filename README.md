# compareimage
Compare two images and detect if they are dealing with the same object.



The aim of this development is to solve the issue of duplicate ads because it has no gain for buyers and sellers. The chosen method is to compare images from ads and to conclude about duplication.
In the development, it is assumed that images from same sellers are compared. No requirements have been made about processing time, even if in production we are looking for the fastest solution.

A first step, not part of the algorithm, has been to organise data. In order to build tests, folders have been created to reproduce different cases : same images, same objects with similar background or takes, same objects with different backgrounds, different objects with similar backgrounds, different objects with different backgrounds.

The development part has been made in Python 3 with well-known libraries : Numpy, Pillow, OpenCV and Scikit-Image. A requirements.txt file contains used libraries with versions.
The command to run the program is : python deduplicate.py path/to/image1 path/to/image2 .

The algorithm that has been created is made of different steps :
1. Preprocessing : Rotation - Remove white bands - Resizing
2. Basic score : Mean Square Error - Structural Similarity
3. Key points matching

They are described below.

# Preprocessing
The first part of the algorithm is related with preprocessing. The aim is to prepare images to be able to successfully process them in the next steps.
## Rotation
In order to obtain a image correctly oriented, metadata of images are used, also called exif.
## White bands removing
The aim of this part is to get an image full of information. Indeed, a picture taken by a user can contain white bands (strips). These bands do not contain any background and represents noisy parts for our similarity algorithm.
A threshold is created to isolate white areas after removing noise. Finally, images are cropped to keep useful information.
## Resizing
In order to correctly compare images, it is important to deal with matrices of same shapes. Objective size is the size of the largest image.

# Basic score
The aim of this part is to create metrics for comparing similarity of images, based on all pixel values.
## Mean Square Error
A first metric is the computation of a basic error, the mean square error.
After few tests, it is quite hard to conclude about similarity with this measure. It has only been used for exact matching.
## Structural Similarity
And then, to conclude about similarity from entire images, the structural similarity can be used. It measures similarity between them not from pixel values but rather from image structure. After tests, a threshold of 0.45 has been chosen.

# Key points matching
Comparing images with basic metrics is not robust, especially for images with different background. So it is important to find another way of comparison.
Stereo vision is the concept of reconstruct an image from two images that have been taken from different viewpoints. To perform this reconstruction, some points from both images are extracted, that contains useful information, and then the best matches between them are selected.
The idea of this step “key points matching” is inspired by stereo vision, especially the points extraction. One of the most known method is called Scale Invariant Feature Transform (SIFT). It uses difference of Gaussian and assigns orientation. It lets us access points that contain more information that the others.
After having found key points, the aim is to obtain matching points between both images. OpenCV includes a method designed for that, Fast Library for Approximate Nearest Neighbors (FLANN). The available code also uses a threshold to only consider close matches.
Finally, a required minimum number of matching key points, 10, has been fixed to ensure similarity between images (found after tests).

# Results
As mentioned above, different kinds of tests have been made. They are sum up here :
same images : 100% correctly classified as same images (2 out of 2)
same objects with similar backgrounds (or takes) : 50% correctly classified (1 out of 2)
same objects with different backgrounds : 66% correctly classified (2 out of 3)
different objects with similar backgrounds : 66% correctly classified (2 out of 3)
different objects with different backgrounds : 100% correctly classified (2 out of 2)

Those results came from twelve different couples of pictures so it not enough to be generalized as a real classifier score. The testing part is very important in the creation of a classifier so it should be more developed. Other situations could be inspected, for example same objects with different colors.

# Further work
## Tuning
As several time mentioned, few thresholds are used. Other fixed parameters are contained in the code because OpenCV methods are full of parameters.
Currently, the tuning has been fastly made but it represents a way of improving results.
## Object of Interest
One of the main problem of this algorithm is the fact that parts of images does contain useful information for similarity testing. 
It could be interesting to first detect in images the object of interest (OOI) before doing any image comparison. Currently, the state of art is Yolo algorithm. It can do multiple object detection and contains many basic labels. It is also possible to create our own object classifier. There are two goals around this idea :
getting labels of object of interest in both images and fastly know if images deal with same category of objects
obtaining coordinates of the OOI in the images. If images deal with same OOI, the algorithm gives us coordinates of the bounding boxes of those objects
Then, it will be possible to remove background or other elements contained in the images that will disrupt the comparison. It will help us to solve the issue around comparing objects in different places.


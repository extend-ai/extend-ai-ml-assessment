
AnamolyDetection - v2 
==============================

This dataset was exported via roboflow.ai 

It includes 21 images.
Knot are annotated in YOLO v4 PyTorch format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Randomly crop between 0 and 20 percent of the image
* Random rotation of between -15 and +15 degrees

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* Random rotation of between -15 and +15 degrees
* Salt and pepper noise was applied to 5 percent of pixels



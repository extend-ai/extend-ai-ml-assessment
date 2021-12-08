# Extend AI Machine Learning Engineer Assessment Solution

The primary method used in this solution is image augmentaion with the albumentations library and transfer learning with weights pre-trained on the Imagenet dataset. The main steps can be broken down into the following:

1. The images are first annotated with Apeer to identify the obvious defects and binary masks are used as part of the training data.
2. Images and corresponding masks are resized and then separated into training and test sets. The test set has only one image. Images from both sets are broken down into small square patches which form the new training and testing data.
3. These patches are then augmented to increase the size of the data set.
4. The U-net architecture from segmentaion models library is trained on the new training data. The model is first loaded with pre-trained weights from segmentaion models library.
5. Training is first carried out for 3 epochs with the encoder weights frozen to fine-tune the decoder and prevent any large updates to the enocder weights. Then the entire model is trained for 100 epochs. The optimizer used is Adam with a learning rate of 0.001.
6. Training and validation splits are generated during training using the model.fit function of Keras api.

The iou_score is used as the performance metric and it stays very low for both training and validation since it is difficult to train on limited annotated data, even with augmentations. The test peformance is similarly very poor and the model fails to segment the obvious defect in the image.

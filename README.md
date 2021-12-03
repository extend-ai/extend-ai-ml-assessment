# Detection_of_Wood_Knots



# Introduction

This repository is created for detecting the wood defects. This is a segmentation problem. Lots of researches have been going on for the segmentation problems including supervised learning like U-Net, and Mask_RCNN for different fields. For this project, supervised learning like U-Net was chosen. Supervised learning was selected because of getting higher accuracy without making a complex project within short-time limit. For following the given instruction, the project was done in Jupyter notebook and pytorch was used. If python file (.py) is needed, I can upload it. 


# Inputs and Labeling

The inputs that were given were unlabeled. For supervise learning, labeling was needed. There are different kinds of knots can be found in woods [1]. For reducing complexity, two labels were chosen and they were normal wood and wood knots. Segmentation masks were created for targets. There were different tools for createing masks such as CVAT and labelme. Labelme was used for this project [2]. Using Labelme, json file was created with label. From the json file, the png images for all segmentation mask were created for targets. COCO format was not used for the limitation of time. After creating segmentaion masks, 14 inputs and 14 targets images were created by rotating 180 degeree each png / jpg file and total of 30 were created.



# Deep learning Model

For this project, I wanted to use both U-Net [3] and Mask R-CNN [4] and compare the differences for detecting the wood knots. In future, I will do that. Here, I used U-Net. U-Net uses the same feature maps that are used for contraction to expand a vector to a segmented image. This would preserve the structural integrity of the image which would reduce distortion enormously. 


# Training and Testing

Data augmentaion was used. Augmentated images were provided for training and validation. For testing, the real images were provided. Losses were plotted and 20 epochs were used.

# Improvement of the Model

1. Use COCO format for the training and testing data 
2. For U-Net model, the loss function and optimizer will be changed to get more accuracy. For further improvement of the model, I can use different normalization and activation layer instead of BatchNormalization and Relu activation.
3. Using data augmentations, more inputs and targets will be created.
4. Use Mask R-CNN instead of U-Net and compare the model.
5. The disadvantage of choosing supervised deep learning for this project was that it relied on the presence of an extensive amount of human-labeled datasets for training. It also needed lots of time. For improvement, an unsupervised model can be created. 
6. Another improvement can be possible by making a semi-supervised learning model. Transfer learning can also be used here. In practice, transfer learning means using as initializations the deep neural network weights learned from a similar task, rather than starting from a random initialization of the weights, and then further training the model on the available labeled data to solve the task at hand.



# References

[1]. Gao, Mingyu, et al. “A Transfer Residual Neural Network Based on ResNet-34 for Detection of Wood Knot Defects.” Forests, vol. 12, no. 2, MDPI AG, 2021, p. 212–, https://doi.org/10.3390/f12020212.

[2]. Available on: https://github.com/wkentaro/labelme/tree/main/examples/tutorial

[3]. Available on: https://towardsdatascience.com/u-net-b229b32b4a71

[4]. Available on: https://github.com/matterport/Mask_RCNN

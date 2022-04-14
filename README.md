# Pose-Estimation-Challenge
Solution for Pose Estimation Challenge 2021 organized by the Advanced Concepts Team of the European Space Agency and the Space Rendezvous Laboratory (SLAB) of Stanford University.

## Description
The competition consists of predicting the position and orientation of the Tango spacecraft in realistic images while only being provided with labels from computer generated examples. The SPEED+ dataset used to train the ML models is composed of 60,000 labeled synthetic images and 9,531 unlabeled real images, and can be downloaded from here: https://purl.stanford.edu/wv398fc4383.

## Approach
1. ResNet50 architecture pre-trained on ImageNet and trained on the SPEED+ dataset to make the first estimate of the pose.
2. Image segmentation with Convolutional Neural Networks (CNNs) to detect the spacecraft.
3. With Camera Matrix, distortion parameters, centroid coordinates and ResNet50 prediction make a second estimate of the position.
4. Project spacecraft into image and validate against the segmented image to find optimum pose.


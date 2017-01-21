# CarND-BehavioralCloning-P3

This work corresponds to Project 3 of Term 1 of the Self-Driving Car Engineer Nanodegree and Udacity.com.

The objective of the project is to write SW for a car to drive autonomously in two circuits of a provided simulator.
Control signals to the simulator include throttle and steering angle.


APPROACH: INTRODUCTION

For the throttle control signal, I implement a PID controller in "drive.py" that mantains speed at 20.
The integral component is needed to fully compensate for drag and resistance forces in the simulator.
The derivative component is introduced to quickly react to changes in slope.

For the steering control signal, I train a Convolutional Neural Network with augmented data from a dataset collected by manually driving the car in the first track of the simulator.

On the remainder of this document, I explain the approach to data collection, preprocessing, split and augmentation, the model architecture and the approach to weight learning and model testing for the steering control signal.


DATA COLLECTION

For steering control signals, I adopt an end-to-end approach with a deep convolutional neural network that takes as input images from a frontal camera and provides as output a real scalar value that corresponds to the steering angle. Note that the CNN implicitly solves the feature detection, path planning and control problems. This approach is very similar to that adopted by NVIDIA with actual cars in https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 

To collect pairs of images and steering angles, I drive several laps in circuit 1 of the simulator with the car centered on the road. I then drive a few laps in the same circuit but in opposite direction. I also record a number of recovery manoeuvres after having had the car intentionally drift towards the left or right side of the road, such that the model learns to recover. Finally, I add the images taken with the left-side and right-side-mounted cameras by Udacity to the dataset, adjusting the steering angle by 0.1 radians. Note that the original dataset is exclusively made of images taken in circuit 1.


DATA PRE-PROCESSING AND SPLIT

...





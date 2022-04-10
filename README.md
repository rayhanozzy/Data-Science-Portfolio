# Data Science Portfolio of Rayhan Ozzy Ertarto

## [Project 1 - Autonomous Driving Application Car Detection](https://github.com/rayhanozzy/Deep-Learning-Specialization-Coursera/blob/main/Course%204:%20Convolutional%20Neural%20Networks/Autonomous_driving_application_Car_detection.ipynb)

- Emerging Tech Subjects: Artificial Intelligence
- Deep Learning Skills: Neural Network
- Machine Learning Methods: Deep Learning
- Back-End Development Skills: TensorFlow
- Programming Languages: Python

- Project description

The purpose of this project:
1) Detected objects in a car detection dataset provided by drive.ai
2) Implemented non-max suppression to achieve better accuracy
3) Implemented intersection over union as a function of non-maximum suppression (NMS)
4) Created usable bounding box tensors from the model's predictions

In this project, I used the algorithm named "You Only Look Once" (YOLO). The algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

Summary for YOLO:
1) Input image (608, 608, 3)
2) The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
3) After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
- Each cell in a 19x19 grid over the input image gives 425 numbers.
- 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
- 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and 80 is the number of classes I'd like to detect

4) Then select only few boxes based on:
- Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
- Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

5) This gives me YOLO's final output.


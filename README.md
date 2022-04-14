# Data Science Portfolio of Rayhan Ozzy Ertarto

## [Project 1: Autonomous Driving Application Car Detection](https://github.com/rayhanozzy/Deep-Learning-Specialization-Coursera/blob/main/Course%204:%20Convolutional%20Neural%20Networks/Autonomous_driving_application_Car_detection.ipynb)

**Emerging Tech Subjects:** Artificial Intelligence

**Deep Learning Skills:** Neural Network

**Machine Learning Methods:** Deep Learning

**Back-End Development Skills:** TensorFlow

**Programming Languages:** Python

**Project Description**

The purpose of this project:

1) Detected objects in a car detection dataset provided by drive.ai

2) Implemented non-max suppression to achieve better accuracy

3) Implemented intersection over union as a function of non-maximum suppression (NMS)

4) Created usable bounding box tensors from the model's predictions

In this project, I used the algorithm named **"You Only Look Once" (YOLO)**. The algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

Summary for YOLO:

1) Input image `(608, 608, 3)`

2) The input image goes through a CNN, resulting in a `(19, 19, 5, 85)` dimensional output.

3) After flattening the last two dimensions, the output is a volume of shape `(19, 19, 425)`:
- Each cell in a `19x19` grid over the input image gives `425` numbers.
- `425 = 5 x 85` because each cell contains predictions for `5` boxes, corresponding to `5` anchor boxes, as seen in lecture.
- `85 = 5 + 80` where `5` is because `(p_c, b_x, b_y, b_h, b_w)` has `5` numbers, and `80` is the number of classes I'd like to detect

4) Then select only few boxes based on:
- Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
- Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

5) This gives me YOLO's final output.

![](/images/image_large.png)

Found 10 boxes for images

## [Project 2: Telecom Customers Churn Prediction](https://github.com/rayhanozzy/Coursera-Project-Network/blob/main/Machine%20Learning%20for%20Telecom%20Customers%20Churn%20Prediction/Machine%20Learning%20Classification%20-%20Telecom%20Customers%20Churn%20Prediction.ipynb)

**Knowledge Representation Skills:** Classification

**Product Management Skills:** Data Analysis

**Data Analytics Skills:** Machine Learning

**Programming Languages:** Python

**Infographic Features:** Data Visualization

**Project Description**

In this project, I train several classification algorithms namely Logistic Regression, Support Vector Machine, K-Nearest Neighbors, and Random Forest Classifier to predict the churn rate of Telecommunication Customers.

Amongst all the trained models, **Random Forest Classifier** algorithm produced the highest area under the ROC curve (AUC).

The following scores are the results of the Random Forest Classifier model
1. Accuracy: `~96%` label accuracy
2. Precision: `~96%` labeled as Retained customers and `~94%` labeled as churned customers
3. Recall: `~99%` labeled as Retained customers and `~76%` labeled as churned customers


![](/images/image_large1.png)

It represents that Random Forest algorithm produced the best AUC.

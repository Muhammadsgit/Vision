## AUTHOR
Muhammad Hassan

## Executive Summary
This project develops an image classifier to categorize bottle images into 5 classes - plastic, beer, soda, water and wine. A convolutional neural network (CNN) model is trained on a dataset of 5000 synthetic bottle images per class with random backgrounds.

Transfer learning with a pre-trained VGG16 model is utilized to leverage robust feature representations and enable training high-quality models from limited data. The model is fine-tuned via the technique of retraining higher layers on the new dataset. The model achieves strong performance, with over 90% accuracy on the test set across all classes. Precision and recall are also high, indicating low misclassification errors.

The classifier generalizes well to real-world bottle images not used in training. This demonstrates its capability to handle synthetic training data and make predictions on real images. In summary, an accurate and robust image classification model is developed through transfer learning. It can categorize bottle images into distinct types, despite training only on synthetic data. This has useful applications in waste sorting, recycling, automated inspection and more.

## Problem Statement
The project aims to develop and evaluate a machine learning model capable of classifying images of bottles into distinct categories based on their types. The dataset comprises synthetically generated images featuring various bottle classes scattered across random backgrounds ([dataset](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset 'Link to dataset')). The current scope of the dataset includes five bottle classes: Plastic Bottles, Beer Bottles, Soda Bottles, Water Bottles, and Wine Bottles. The goal is to accurately categorize each image into one of these predefined classes.

This task addresses the challenges inherent in computer vision and image classification, particularly in dealing with **synthetic/augmented imagery and diverse backgrounds**. The project involves exploring and applying advanced machine learning techniques, including convolutional neural networks (CNNs) and transfer learning, to achieve high accuracy in classification. 


## About



### File Directory / Table of Contents
This is an alphabetical list of the repository's directory and file structure.

- README.md
- code
  - 01_cleaning.ipynb
  - 02_eda.ipynb
  - 03_
  - 04_model
  - 05_model_

- datasets
    - 
- images
 
- pickles

- presentation
  - 
- results
  
   
### Software Requirements

Jupyter Notebook
Matplotlib.pyplot
NumPy
Pandas
Scikit-Learn (sklearn)

### Data Classes


### Feature	Description	Details


### Conclusions and Recommendations


### Findings:


### Recommendations:


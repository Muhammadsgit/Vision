
## Executive Summary
This project develops an image classifier to categorize bottle images into 5 classes - plastic, beer, soda, water and wine. A convolutional neural network (CNN) model is trained on a dataset of 5000 synthetic bottle images per class with random backgrounds.

Transfer learning with a pre-trained VGG16 model is utilized to leverage robust feature representations and enable training high-quality models from limited data. The model is fine-tuned via the technique of retraining higher layers on the new dataset. The model achieves strong performance, with over 90% accuracy on the test set across all classes. Precision and recall are also high, indicating low misclassification errors.

The classifier generalizes well to real-world bottle images not used in training. This demonstrates its capability to handle synthetic training data and make predictions on real images. In summary, an accurate and robust image classification model is developed through transfer learning. It can categorize bottle images into distinct types, despite training only on synthetic data. This has useful applications in waste sorting, recycling, automated inspection and more.

## Problem Statement
Categorizing waste bottles by material type is an important task for recycling and waste management. However, training image classification models that are robust to real-world bottle images is challenging, particularly when limited to synthetic training data.

The key problem is developing an image classifier capable of accurately categorizing bottle images into distinct types (plastic, glass, etc.) despite being trained only on synthetically generated data. The model must handle variations of bottle appearance, lighting, angle and background in real-world images.

This project aims to address this problem by leveraging deep transfer learning techniques. A convolutional neural network pre-trained on large-scale data is fine-tuned on synthetic bottle images to adapt its feature extraction capabilities.

The goal is to train an accurate multi-class classifier for bottle images that generalizes well to real-world data. The target is over 90% precision and recall across all classes using only synthetic images for training.

Success would demonstrate an ability to cope with synthetic training data and make predictions on real images. This has significant applications in waste and recycling management, automated inspection systems, assistance technology and more.

The key challenges include limited training data, class imbalance, and domain shift between synthetic and real-world images. State-of-the-art deep learning techniques will be explored to tackle these challenges.


### File Directory / Table of Contents
This is an alphabetical list of the repository's directory and file structure.

.  
├── README.md  
├── code/  
│   └── Bottle_Image_classifier.ipynb  
├── datasets/  
│   └── link.txt  
├── saved_models/
│   └──  
├── presentation/  
│   └──   
├── results/  
│   ├── confusion_matrix.png  
│   └── accuracy_plot.png  
└── requirement.txt  
   
### Data Classes


### Feature	Description	Details


### Conclusions and Recommendations


### Findings:


### Recommendations:


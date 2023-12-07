
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
The dataset used to fine-tune our bottle image classifier consist of 5 classes namely, Plastic Bottles , Beer Bottles, Soda Bottles, Water Bottles, and Wine Bottles. 

Plastic Bottles:

![00000000](https://github.com/Muhammadsgit/Vision/assets/17506063/1ca43d16-577f-4f63-b216-f1cbd4c34bc9)

Beer Bottles:

![00000182](https://github.com/Muhammadsgit/Vision/assets/17506063/8aaa4b90-d65b-43ac-b331-11d56d177531)

Soda Bottles:

![00000033](https://github.com/Muhammadsgit/Vision/assets/17506063/8687c81a-cfe1-408c-8b9c-77a634b8ad98)

Water Bottles:

![00000138](https://github.com/Muhammadsgit/Vision/assets/17506063/b0a01214-febc-43b4-a90b-3c8a9a34f7e0)

Wine Bottles:

![00000205](https://github.com/Muhammadsgit/Vision/assets/17506063/f4dc98af-cda1-49bd-8068-f1dc98db2c95)

### Model Architecture
This image classification project aims to categorize bottle images into five distinct classes: plastic, beer, soda, water, and wine. To achieve this, a convolutional neural network (CNN) model is developed, trained on a dataset of 25,000 synthetic bottle images (5,000 per class) with varied backgrounds.

The model architecture leverages transfer learning, employing a pre-trained VGG16 model as the foundational layer. VGG16, known for its effectiveness in image recognition tasks, provides a robust feature representation base. This approach is particularly beneficial when working with a limited dataset, as it allows the model to utilize features learned from a much larger dataset (like ImageNet).

After integrating the VGG16 base, the model undergoes fine-tuning where higher layers are retrained on the project's specific dataset of synthetic bottle images. This fine-tuning process adapts the model to the nuances of the bottle classification task, leading to an impressive performance with over 90% accuracy on the test set across all classes. High precision and recall values further indicate a low rate of misclassification errors, which is crucial for practical applications.

![Model arc](https://github.com/Muhammadsgit/Vision/assets/17506063/1542d43b-05e5-4fd9-8652-26c42b37a977)

One of the notable achievements of this classifier is its ability to generalize effectively to real-world bottle images, despite being trained only on synthetic data. This generalization capability is essential, demonstrating that the model can reliably handle and make accurate predictions on real images it has not encountered during training.



###  Conclusions and Recommendations

In conclusion, this project successfully developed an accurate image classifier capable of categorizing synthetic bottle images into 5 distinct types with over 90% precision and recall. The model generalizes well to real-world bottles, despite training only on synthetic data.

The use of deep transfer learning proved highly effective, enabling the model to learn robust feature representations from limited training data. Fine-tuning a pre-trained VGG16 model on the new dataset allowed adaptation of features to the bottle classification task.

The model is ready to be integrated into a waste sorting and recycling system. It can identify bottle material types with high reliability, enabling automated sorting and treatment.

For future work, it is recommended to expand the model to additional bottle categories such as glass, cartons, pouches etc. Augmenting the synthetic training data with a small set of real images via techniques like mixup could further enhance generalization.

The model should be deployed on an embedded system in the recycling plant and evaluated on a live input stream. With enhancements, this technology can be scaled and commercialized to automate waste bottle sorting for sustainability and business benefits.



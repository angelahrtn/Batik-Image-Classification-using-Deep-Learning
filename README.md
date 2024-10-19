# Batik-Image-Classification
# Project Overview
This project, titled "Batik Image Classification using Deep Learning", was completed as part of a group assignment for the Deep Learning course in the 4th semester. The primary goal of the project is to classify Indonesian batik motifs using deep learning techniques. The project explores both custom-built models (scratch models) and transfer learning techniques to classify images of batik into one of three motifs: Sidoluhur, Tambal, and Betawi. We implemented and compared several models, including CNN-based architectures from scratch and transfer learning using EfficientNet and MobileNet. The project was developed in Python using PyTorch and involved preprocessing the dataset, training models, and evaluating their performance on a test set.

# Case Description
The case focuses on automating the classification of batik images, a task that is traditionally done manually by experts. Given the complexity of batik patterns, this process is time-consuming and prone to error. The dataset used consists of 50 images for each class, selected from a broader collection of batik motifs. Our task was to create models that could automatically classify these images into the appropriate batik motif. This involves leveraging deep learning techniques to build and optimize models that can generalize well to unseen data.

# Objectives
- Develop and compare models for batik image classification using both scratch models and transfer learning techniques.
- Optimize model performance through hyperparameter tuning, including dropout layers and learning rate adjustments.
- Evaluate and compare model accuracy to determine which approach (scratch models vs. transfer learning) yields the best performance.
- Deploy the final model for practical use in a batik classification application.

# Project and Steps
1. Data Preprocessing:
- The batik dataset was sourced from Kaggle and includes images resized to 224x224 pixels for uniformity.
- Data augmentation was applied (random horizontal flip) to increase dataset variability and reduce overfitting.
- Images were normalized using the standard ImageNet normalization values.

2. Model Development:
- Custom Scratch Models: We designed CNN-based models with various layers including convolutional layers, pooling layers, and fully connected layers.
- Transfer Learning Models: EfficientNet and MobileNet architectures were used with pre-trained weights. We fine-tuned the final classification layers to adapt to the three batik motif classes.

3. Training:
- Both sets of models were trained using an 80-10-10 train-validation-test split.
- Hyperparameter tuning was performed, including adjusting the number of epochs, learning rate, and dropout rates to improve performance.

4. Evaluation:
- The models were evaluated based on accuracy, precision, recall, and F1 score.
- The best-performing model was selected based on the highest accuracy on the test data.

# Tools
- Python: Core programming language for model implementation.
- Libraries:
  - PyTorch: For building and training the models.
  - EfficientNet and MobileNet: For transfer learning-based classification models.
  - Google Colab: For model training and development.
  - Matplotlib & Seaborn: For data visualization and model performance analysis.
  - Pickle: For saving and loading trained models.

# Challenges
- Overfitting: In both the scratch models and transfer learning models, overfitting was a persistent issue. We experimented with dropout layers and data augmentation to mitigate this but still observed fluctuations in validation accuracy.
- Limited Dataset: The relatively small number of images (50 per class) posed a challenge in achieving high generalization accuracy. Data augmentation and transfer learning were employed to address this.
- Training Time: Training deep learning models on a dataset with high variability in patterns like batik can be computationally expensive, especially for scratch models.

# Conclusion
After evaluating the performance of both the scratch models and transfer learning models, FixEfficientNet-B2 original  provided the best performance with a testing accuracy of 86.67%. Among the scratch models, model 1 provided the best performance achieved was 80% testing accuracy using a CNN model with dropout and an increased number of epochs.

The project demonstrated that transfer learning, despite requiring less time and computational resources, can provide competitive results compared to scratch-built models. However, custom-built models still showed potential for improvement with additional training time and hyperparameter tuning.

# Machine Learning - Image Classification with TensorFlow
Deep Neural Networks classification of handwritten digits using the MNIST dataset.

## Results
**Final Model - Test Accuracy: 99.4%**

<img src="assets/img/model-comparisons.png">

## DNN Architecture and ML Workflow

### Overview
1) Preprocessing was performed on the dataset of 70,000 images.
2) Fully-connected network baseline model built.
3) Hyperparameter tuning performed on fully-connected network using Keras Tuner's Hyperband.
4) Convolutional neural network (CNN) baseline model built.
5) Bayesian Optimization CNN hyperparameter tuning.
6) Data augmentation on training dataset.
7) Final model chosen and evaluated on test dataset.

### 1) Preprocessing
- 70,000 28x28x1 images of handwritten digits 0-9, split into 50k/10k/10k train/validation/test datasets, with additional data augmentation on the training set performed later in the model-improvement process.
- Pixel intensity values of integers between 0-255 were scaled to floats between 0-1.

### 2) Fully-Connected Network - Baseline Model
- Input layer: 28x28x1 flattened to 1x784.
- Multiple fully-connected hidden layers. For baseline model, used 2 sets of dense+dropout hidden layers were used, each with 64 neurons and ReLU activation function.
- Output layer of size 10, representing 10 different digit classses. Softmax activation function connecting to output layer.
- Adaptive Moment Estimation used for optimizer and Sparse Categorical Crossentropy used for loss.
- Early stopping implemented to prevent overfitting. Stopped when validation loss began increasing with a patience factor of 5 and restore_best_weights=True (i.e., once val_loss increased for 5 consecutive epochs, stop training and "roll back" by 5 epochs).

### 3) Fully-Connected Network - Hyperparameter Tuning
- Tuned the following hyperparameters with Keras Tuner's Hyperband.
  - num_hidden_layers = [1, 2, 3] # Number of dense layers. Each dense layer also followed by a dropout layer.
  - hidden_layer_size = [32, 64, 128, 256] # Number of neurons in each dense layer.
  - activation = ['relu', 'tanh']
  - dropout_rate = [0.1 - 0.5 with step=0.05]
  - l2_strength = [0.0001, 0.001, 0.01] # L2 regularization (ridge regression) applied to each dense layer.
  - learning_rate = [0.0001 - 0.01 with step=0.001]
 
<img src="assets/img/training-curve-dense-hpt.png">
    
- The best Hyperband model was re-trained with the best-performing hyperparameter combination, and allowed to train with a patience=10 (instead of the patience=5 used in Hyperband).
- **Best Fully-Connected Network Architecture: Results After 92 Epochs:**
  - Accuracy: Train=99.30% | Val=98.15%
  - Loss: Train=0.0471 | Val=0.0836
 
### 4)
### 5)
### 6)
### 7)
 
- Conclusion: The highest validation accuracy from all hyperparameter combinations was 99.88%, which only had a 0.03% difference from the training accuracy; therefore this was considered the best model. Finally, the test dataset was utilized to evaluate the model associated with this validation accuracy; this final model achieved a 98.01% accuracy on the test dataset.
  - Test accuracy: 98.01% | Validation accuracy: 99.88% | Train accuracy: 99.85%
  - Epochs trained (until overfitting began): 23 | Training time: 0min 28sec
  - Best Hyperparameters: {'hidden_layer_size': 128, 'hidden_layer_size_variation': 'constant', 'hidden_layer_depth': 2, 'activation_function': 'tanh'}

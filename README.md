# DEEP-LEARNING-PROJECT

COMPANY:CODETECH IT SOLUTIONS

NAME:MAHVEEN SULTANA

INTERN ID:CT04DZ47

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

ENTER DESCRIPTION OF TASK:As part of CodTech IT Solutions Internship Task-2, I implemented a Deep Learning model for Image Classification using the PyTorch framework. This project focused on recognizing handwritten digits from the MNIST dataset, which is a standard benchmark dataset widely used in deep learning research.

The main goal of this task was to build a deep learning model, train it on a real dataset, evaluate its performance, and visualize the results.

Tools and Technologies Used

1. Programming Language:
Python 3.11 was used for coding because of its simplicity and rich ecosystem of libraries for deep learning.

2. Deep Learning Framework:
PyTorch was chosen for building and training the model. It provides dynamic computation graphs and is very beginner-friendly.

3. Dataset Loader & Transformations:
Torchvision was used to directly download and preprocess the MNIST dataset with simple transformations.

4. Visualization Library:
Matplotlib was used to plot the training loss curve and observe model learning progress.

5. Development Environment / Editor:
Visual Studio Code (VS Code) was used as the primary editor for writing and running the code.
It provided syntax highlighting, debugging tools, and an integrated terminal for seamless workflow.
6. Hardware Configuration:
The code is written to automatically use GPU (CUDA) if available. In my case, the model was trained on CPU, but the same code can run on GPU for faster training.

Project Workflow

1️⃣ Dataset Preparation
I used the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0–9). Each image is 28x28 pixels, grayscale.
Using Torchvision transforms, the dataset was converted into tensors and normalized to a range of [-1, 1], which helps the neural network converge faster.
The dataset was split into train_loader and test_loader with a batch size of 64 for mini-batch gradient descent.

2️⃣ Model Architecture
I implemented a Simple Fully Connected Neural Network (Feedforward Neural Network) with the following structure:
Input Layer: 784 neurons (flattened 28x28 image)
Hidden Layer 1: 128 neurons + ReLU activation
Hidden Layer 2: 64 neurons + ReLU activation
Output Layer: 10 neurons (one for each digit class)

3️⃣ Training Setup
Loss Function: CrossEntropyLoss (for multi-class classification)
Optimizer: Adam optimizer with a learning rate of 0.001
The model was trained for 3 epochs over the training dataset.
During training, for each epoch, the model performed forward propagation, computed the loss, performed backpropagation, and updated weights using the optimizer.

4️⃣ Evaluation
After training, the model was tested on 10,000 test images. The model predicted the digits with a test accuracy of ~98%, showing strong performance even with a simple network.

5️⃣ Visualization
To analyze learning behavior, I plotted the Training Loss Curve using Matplotlib. The graph showed a decreasing trend in loss with each epoch, indicating that the model was learning effectively.


Applications of This Project

This project demonstrates image classification, which has several real-world applications:

Handwriting recognition (digitizing handwritten forms)
Banking applications (automatic cheque reading)
Postal services (recognizing handwritten postal codes)
OCR (Optical Character Recognition) systems
Document scanning & digitization
Additionally, this serves as a foundation for more advanced deep learning models, such as Convolutional Neural Networks (CNNs), which achieve even higher accuracy for image-based tasks.


Editor Platform & Execution:

I used Visual Studio Code as my main editor.
Installed required libraries (torch, torchvision, matplotlib) using pip install.
Wrote the code in a file named deeplearning.py.
Ran the code using the integrated terminal inside VS Code.
The code automatically downloaded the MNIST dataset, trained the model, printed epoch-wise loss values, final test accuracy, and displayed the loss graph.

OUTPUT:
<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/afab1410-1f08-45db-906a-09c769d3b481" />

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/4bca7bdd-cefe-40d0-bfb6-6ce770bfc98d" />

<img width="1366" height="768" alt="Image" src="https://github.com/user-attachments/assets/efc32ecf-7a18-464d-8ec0-356933a18069" />

# Pneumonia Detection Using X-Ray Images with Deep Learning

## Domain
**Artificial Intelligence, Medical Imaging**

## Sub-Domain
**Computer Vision, Image Classification**

## Techniques
- Deep Learning
- Transfer Learning
- Custom CNN Architecture

## Applications
- Healthcare Diagnostics
- Pneumonia Detection

---

## Project Summary
This project demonstrates the use of advanced deep learning techniques to detect pneumonia from chest X-ray images. It combines transfer learning with a fine-tuned "InceptionV3" model and a custom-built convolutional neural network (CNN) to achieve high performance in binary image classification tasks (Pneumonia vs. Normal).

---

## Highlights

- **Data Preparation**: A comprehensive dataset of 5,856 X-ray images (~1.15GB) was used, split into training, validation, and testing sets.
- **Transfer Learning**: Leveraged the pre-trained "InceptionV3" model, freezing early layers and fine-tuning subsequent ones to adapt to the new classification problem.
- **Custom CNN Performance**: Achieved notable metrics, including a testing accuracy of 89.53% and a recall of 95.48% for pneumonia cases.

---

## Dataset Information

**Dataset**: Chest X-Ray Images (Pneumonia)

### Source
- [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2)

### Referenced Research
- [Daniel S. Kermany et al., 2018](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

### Dataset Breakdown
- **Training Set**: 5,216 images (~1.07GB)
- **Validation Set**: 320 images (~42.8MB)
- **Testing Set**: 320 images (~35.4MB)

---

## Model Details

### Base Model
Pre-trained InceptionV3 (ImageNet weights)

### Custom Architecture
Deep Convolutional Neural Network for Pneumonia Detection

### Frameworks and Libraries
- TensorFlow
- Keras

### Training Parameters for Custom CNN
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 30
- **Training Time**: ~2 hours

---

## Performance Metrics

- **Testing Accuracy**: 89.53%
- **Loss**: 0.41
- **Precision**: 88.37%
- **Recall**: 95.48% (for pneumonia cases)

---

## Tools and Environment

- **Programming Language**: Python
- **Development Environment**: Anaconda
- **Libraries**: Keras, TensorFlow, InceptionV3, OpenCV

---

## Repository

Explore the full code and implementation here:  
**[Pneumonia Detection using X-Ray Images with Deep Learning](https://github.com/Samarthcoder01/Pneumonia-Detection-using-X-Ray-images-with-Deep-Learning)**

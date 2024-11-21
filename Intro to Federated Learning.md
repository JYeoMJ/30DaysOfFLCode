# Federated Learning: Short Course Notes from DeepLearning.AI

## Framework: Flower

- **Objective**: Run distributed machine learning (ML) training jobs while enhancing data privacy.
- **Use Case**: Train models (e.g., medical imaging) across distributed data sources without centralizing sensitive data (e.g., hospital images restricted by regulations).

## Core Concepts

- **Federated Learning (FL)**: A method to train ML models on decentralized data sources. 
  - Instead of moving data to the training process, move the training process to the data.
  - Centralizes model parameters, not raw data, ensuring privacy.
- **Privacy Advantage**: 
  - Ideal for sensitive domains like healthcare.
  - Complies with privacy regulations while leveraging diverse data.

## Federated Training Process

1. Distribute training jobs to local data sources.
2. Train models locally on each source.
3. Aggregate model parameters at a central server.
4. Update global model without exposing raw data.

## Key Topics Covered

- **System Tuning**: Techniques to optimize federated systems for performance and efficiency.
- **Data Privacy**:
  - **Differential Privacy (DP)**: Protects individual data points (e.g., images or messages) from being inferred.
- **Bandwidth Management**: Strategies to minimize communication overhead in FL setups.

---

This provides a high-level understanding of federated learning and its relevance, especially in privacy-sensitive domains.

---

## Setup: Requirements

### `requirements.txt`

```
# Python version: 3.9.6
# Dependencies

flwr==1.10.0
ray==2.6.3
flwr-datasets[vision]==0.2.0
torch==2.2.1
torchvision==0.17.1
matplotlib==3.8.3
scikit-learn==1.4.2
seaborn==0.13.2
ipywidgets==8.1.2
transformers==4.42.4
accelerate==0.30.0
```
---

## Utilities Code

### Utility Functions

- **Transformations**:
  - Normalize and convert datasets to tensors.
- **Model Definition**:
  - `SimpleModel`: A simple neural network for MNIST digit classification.
- **Training**:
  - `train_model`: Train the model on a dataset using SGD optimizer and CrossEntropy loss.
- **Evaluation**:
  - `evaluate_model`: Evaluate the model on a test dataset and return accuracy and loss.
- **Dataset Filtering**:
  - `include_digits`: Retain only specified digits in a dataset.
  - `exclude_digits`: Exclude specified digits from a dataset.
- **Visualization**:
  - `plot_distribution`: Visualize digit distribution in a dataset.
  - `plot_confusion_matrix`: Visualize confusion matrix as a heatmap.

### Example Code
```python
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x
```

---

## Main Workflow

### Data Preparation
1. **Dataset**:
   - Use MNIST dataset (training and test).
   - Split training dataset into 3 parts for simulating decentralized data.
   - Exclude specific digits from each split to simulate incomplete data distribution.

2. **Visualization**:
   - Plot the distribution of digits in each dataset split using `plot_distribution`.

### Example Code
```python
# Import MNIST dataset
from torchvision import datasets

trainset = datasets.MNIST("./MNIST_data/", download=True, train=True, transform=transform)

# Split dataset into 3 parts
split_size = len(trainset) // 3
torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_size] * 3)

# Exclude specific digits
part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

# Visualize dataset distribution
plot_distribution(part1, "Part 1")
plot_distribution(part2, "Part 2")
plot_distribution(part3, "Part 3")
```

---

### Model Training and Evaluation
1. **Training**:
   - Train separate `SimpleModel` instances (`model1`, `model2`, `model3`) on each dataset split using `train_model`.

2. **Evaluation**:
   - Evaluate each model on:
     - Full test dataset.
     - Subset of the test dataset containing relevant digits only.

3. **Accuracy Reports**:
   - Display accuracy for full and subset test datasets.

4. **Confusion Matrix**:
   - Compute confusion matrices for all test data using `compute_confusion_matrix`.
   - Visualize confusion matrices using `plot_confusion_matrix`.

### Example Code
```python
# Train models on different datasets
model1 = SimpleModel()
train_model(model1, part1)

model2 = SimpleModel()
train_model(model2, part2)

model3 = SimpleModel()
train_model(model3, part3)

# Testset Preparation
testset = datasets.MNIST("./MNIST_data/", download=True, train=False, transform=transform)

# Evaluate models
_, accuracy1 = evaluate_model(model1, testset)
_, accuracy1_on_137 = evaluate_model(model1, include_digits(testset, [1, 3, 7]))
print(f"Model 1 Accuracy on all: {accuracy1:.4f}, [1,3,7]: {accuracy1_on_137:.4f}")

# Analyze confusion matrices
confusion_matrix_model1 = compute_confusion_matrix(model1, testset)
plot_confusion_matrix(confusion_matrix_model1, "Model 1 Confusion Matrix")
```
---

## Key Insights
- Models trained on incomplete datasets perform well on subsets but may generalize poorly to the entire dataset.
- Federated learning frameworks (e.g., Flower) simulate decentralized, privacy-preserving training processes.
- Visualizations and evaluations highlight the challenges of incomplete data distribution in federated systems.

---



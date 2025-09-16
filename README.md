# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

This experiment demonstrates transfer learning using a pre-trained ResNet18 model on a custom image dataset. Instead of training a deep neural network from scratch, the pre-trained model’s feature extraction layers are reused, and only the final classification layer is retrained. This approach reduces training time, requires less data, and achieves high accuracy.
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:Data Preprocessing – Resize all images to 224×224 and convert them into tensors suitable for ResNet input.

### STEP 2: Dataset Loading – Organize images into train/test sets and load them using ImageFolder and DataLoader.

### STEP 3:Load Pretrained Model – Use ResNet18 trained on ImageNet as the base model.

### STEP 4:Modify Final Layer – Freeze earlier layers and replace the fully connected layer to match the number of dataset classes.

### STEP 5:Train and Evaluate – Train only the final layer, then test the model and analyze results using a confusion matrix and classification report.
<br/>

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.resnet18(pretrained=True)


# Modify the final fully connected layer to match the dataset classes

for param in model.parameters():
    param.requires_grad = False   # freeze earlier layers

model.fc = nn.Linear(model.fc.in_features, num_classes)


# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)



# Train the model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        ...



```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="700" height="547" alt="image" src="https://github.com/user-attachments/assets/d1f01bb5-8b3e-45f9-baf6-0b58af6bce99" />

</br>
</br>
</br>

### Confusion Matrix
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/471daceb-7b17-48ea-bf8c-7d00826c715f" />

</br>
</br>
</br>

### Classification Report
<img width="593" height="236" alt="image" src="https://github.com/user-attachments/assets/b0d1f67e-f8cf-4b2b-862a-fddeea25ff27" />

</br>
</br>
</br>

### New Sample Prediction
<img width="378" height="431" alt="image" src="https://github.com/user-attachments/assets/7ad41571-4952-488a-bcba-7d2539a74a20" />

</br>
</br>

## RESULT
</br>
</br>
</br>

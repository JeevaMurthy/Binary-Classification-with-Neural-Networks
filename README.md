# Binary Classification with Neural Networks (PyTorch)

## AIM  
To develop a neural network model for tabular data using PyTorch, capable of handling both categorical and continuous features, and predicting income levels (<=50K or >50K) from the given dataset.  

---

## THEORY  
Regression and classification problems involve predicting output variables based on input features.  

- **Regression** → Predicting continuous values  
- **Classification** → Predicting categorical values  

Traditional linear/logistic regression often struggles with complex feature interactions. Neural networks can capture these relationships by:  
- Using **embedding layers** for categorical features  
- Applying **batch normalization** for continuous inputs  
- Employing **non-linear activations** (ReLU) and **dropout** for regularization  
- Learning via **backpropagation** with an optimizer (Adam/SGD)  

In this project, a **feedforward neural network** is built that combines categorical embeddings and continuous variables to predict income categories.  

---

## Neural Network Architecture  

```text
TabularModel(
  (embeds): ModuleList(Embeddings for categorical cols)
  (emb_drop): Dropout(0.4)
  (bn_cont): BatchNorm1d(2)
  (layers): Sequential(
    Linear(22 → 50) → ReLU → BatchNorm1d → Dropout  
    Linear(50 → 2)
  )
)
```
## Design Steps  

### STEP 1: Generate / Load Dataset  
- Load the dataset `income.csv` (30,000 rows × 10 columns).  
- Identify categorical (`sex`, `education`, `marital-status`, `workclass`, `occupation`), continuous (`age`, `hours-per-week`), and target (`label`) columns.  

### STEP 2: Initialize the Neural Network Model  
- Define a custom neural network class `TabularModel`.  
- Use **embedding layers** for categorical variables.  
- Apply **batch normalization** for continuous variables.  
- Combine both categorical embeddings and continuous features.  


### STEP 3: Define Loss Function and Optimizer  
- Use **CrossEntropyLoss** as the loss function for classification.  
- Optimize parameters using **Adam optimizer** with a learning rate of `0.001`.  


### STEP 4: Train the Model  
- Train for **300 epochs**.  
- Forward pass → Compute predictions.  
- Calculate loss → Backpropagate errors.  
- Update weights using optimizer.  
- Record loss for visualization.  


### STEP 5: Plot the Loss Curve  
- Convert loss values to NumPy array.  
- Plot loss across epochs using `matplotlib`.  
- Check if loss decreases steadily (indicating convergence).  


### STEP 6: Visualize the Best-Fit Model  
- Compare the predicted results with actual labels.  
- Plot test dataset predictions vs actual values.  
- Highlight learned decision boundaries / classification line.  


### STEP 7: Make Predictions  
- Use the trained model to predict on unseen test data.  
- Calculate **test loss** and **accuracy**.  
- Print evaluation metrics (Accuracy ~88%).  

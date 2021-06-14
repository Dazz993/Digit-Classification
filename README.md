# Digit-Classification
Digit Classification and Model Analysis (CS385 Final Project1)

## Dataset

Dataset: The SVHN Database of handwritten digits. 

Download link: http://ufldl.stanford.edu/housenumbers/.

## Objective

- Model-level
  - How to understand and implement different models?
  - What are the advantages and disadvantages of different models
- Feature-level
  - Visualize features
  
## Models

### Logistic Regression

1. Implemented with pytorch `nn.Linear` and `nn.CrossEntropyLoss`.
2. `lr = 0.0001`: best test loss = 2.27954555, best test accuracy = 0.25487861
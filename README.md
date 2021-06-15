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
  
## Feature Extraction Methods

### SIFT (complex). 

Reference: [pythonSIFT](https://github.com/rmislam/PythonSIFT).


### HOG. 

Reference: [Wikipedia-HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients).

1. Procedure:
    1. Gradient computation
    2. Orientation binning
    3. Descriptor blocks
    4. Block normalization

## Models

### Linear Model Crossentropy

1. Implemented with pytorch `nn.Linear` and `nn.CrossEntropyLoss`.
2. `lr = 0.0001`: best test loss = 2.27954555, best test accuracy = 0.25487861

```shell
# original data
python main_linear_model_crossentropy.py --cfg configs/linear_model_crossentropy.yaml |& tee -a docs/logs/LinearModelCrossEntropy.log

# HOG feature extraction
python main_linear_model_crossentropy.py --cfg configs/linear_model_crossentropy_hog.yaml |& tee -a docs/logs/LinearModelCrossEntropyWithHOG.log 
```

### Logistic Regression 10 Binary Classifier

1. Implemented with pytorch `nn.Linear` and negative log-likelihood losses.
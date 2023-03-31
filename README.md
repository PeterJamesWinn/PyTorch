
linear_regression.ipynb
Example based on https://www.learnpytorch.io/01_pytorch_workflow/
uses nn.Module as base class and defines own weights

linear_regression_1.ipynb
Example based on https://www.learnpytorch.io/01_pytorch_workflow/ 
- compared to linear_regression.ipynb this uses the nn.linear() 
class for the linear regression and looks at moving tensors between 
devices, including creating device agnostice code - i.e. to take a gpu 
if present but run on cpu if not. 

linear_regression_data_loader.ipynb
Linear regression example with torch's DataLoader
taken from: https://machinelearningmastery.com/using-optimizers-from-pytorch/
with nothing currently added by me. 

loading_regression_model2_parameters.ipynb
example of loading parameters stored in the models subdirectory

models
subdirectory containing saved model parameters.


colab_binary_classification_default_initialisations.ipynb
loosely following https://www.learnpytorch.io/02_pytorch_classification/
with help from:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
and Andrej Karpathy's youtube channel. 
Use pytorch's default initialisations for an nn.Linear layer. 
See also: colab_binary_classification_alternative_initialisations.ipynb

colab_binary_classification_alternative_initialisations.ipynb
loosely following https://www.learnpytorch.io/02_pytorch_classification/
with help from:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
and Andrej Karpathy's youtube channel. 
This version uses Kaiming He /Xavier type scaling of the normal 
distribution for initialisation and zero bias. 
Compared to the two identical instances of the model with default 
initialisations, this initialisation
converges the model to 100% accuracy on train and validation with 
2/3 to a half the number of epochs, 
and the model trained for the same number of steps has a much 
smoother decision boundary.
See also: colab_binary_classification_default_initialisations.ipynb 

colab_multiclass_classification_mini_batch_better_init.ipynb
Multiclass classification
loosely following https://www.learnpytorch.io/02_pytorch_classification/
with help from:
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
and Andrej Karpathy's youtube channel. 
Has an example with full batch and one with mini batches. 
compared to the binary training examples that I implemented this has better
initialization of the output layer, to give more even probability predictions
on initialization. 

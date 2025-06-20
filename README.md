# scm-research
Repository for research on the simple connectome model. Using a non-linear model to predict genetic rules governing synaptic formation. 

## File Structure Overview
- The "data" folder contains the raw biological data used for the models
- The "src" folder contains the nonlinear model code and analysis
- The "output" folder contains the model predictions, statistics, and other figures

## Using The Nonlinear Model
The scm_optim.py file contains the code for model prediction. Use the predict_O function is used to get the predicted genetic rules given the connectome, genetic expression matrix, and contactome. Set the gradient_func parameter to one of the gradient functions (also defined in scm_optim) to specify the desired non-linearity. 


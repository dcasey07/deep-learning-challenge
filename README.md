# Deep Learning Challenge: Module 21

## Objective

The goal for the module was to create a deep learning neural network that can aid the non profit foundation Alphabet Soup in the selection of funding applicants based on the likelihood that they will have a successful venture.

## Dependencies and Implementation
This is performed in Python using Google colab notebooks, Pandas and Tensorflow, along with `train_test_split` and `StandardScaler` modules from the sklearn library, as well as the `ModelCheckpoint` module from the keras.callbacks library.

## Analysis Report

The data used for this analysis is a combination of categorical and numerical features that are as follows:
- `EIN and NAME`: Identification columns
- `APPLICATION_TYPE`: Alphabet Soup application type
- `AFFILIATION`: Affiliated sector of industry
- `CLASSIFICATION`: Government organization classification
- `USE_CASE`: Use case for funding
- `ORGANIZATION`: Organization type
- `STATUS`: Active status
- `INCOME_AMT`: Income classification
- `SPECIAL_CONSIDERATIONS`: Special considerations for application
- `ASK_AMT`: Funding amount requested
- `IS_SUCCESSFUL`: Was the money used effectively

### Data Preprocessing and Cleaning

**What variable(s) are the target(s) for your model?**
- `IS_SUCCESSFUL` is the target for this model, which is a binary classification of 0 for unsuccessful and 1 for successful.

**What variable(s) are the features for your model?**
- `APPLICATION_TYPE`, `APPLICATION_TYPE`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT` were the features of this model.
  - `APPLICATION_TYPE` data was binned and sorted so that all application types in the dataset below 500 were rebranded as "Other". This "Other" label makes up only 276 total entries out of the 30,000+ entries present in the data.
  - `CLASSIFICATION` column was similarly binned and a threshold for classifications below a 1,000 were rebranded as "Other", which resulted in the "Other" label containing 2,261 entries.
  - The categorical data was processed with `pd.get_dummies()` in order to convert them to numerical data for the model.

**What variable(s) should be removed from the input data because they are neither targets nor features?**
- Only the `EIN` and `NAME` columns were dropped as viable features, as they are purely for identification purposes.

### Model Compilation, Training, and Evaluation

**How many neurons, layers, and activation functions did you select for your neural network model, and why?**

**Were you able to achieve the target model performance?**

**What steps did you take in your attempts to increase model performance?**


### Model Optimization



### Summary


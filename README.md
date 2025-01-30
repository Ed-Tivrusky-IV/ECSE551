# ECSE551 Assignments
## A1
The goal of this homework is to gain experience in implementing the well-known linear classifier logistic regression, from scratch, as are discussed in the class lectures. You are required to perform two-class classification on two datasets, CKD and Battery, by using Logistic Regression classifier. The datasets are attached.

Chronic Kidney Disease (CKD): This dataset comprises 28 numerical features, each representing a specific biological measurement related to a patient, such as creatinine levels, glomerular filtration rate (GFR), and urine protein levels. The 29th column serves as the target variable, indicating whether the patient is classified as 'Normal' or diagnosed with CKD ('CKD'). The dataset is commonly used for binary classification tasks, where the objective is to predict the presence or absence of CKD. Each row corresponds to a unique patient, with the last column providing the ground truth class label ('Normal' or 'CKD'). It is important to note that while some features are normalized, others remain in their original scale.

Battery Dataset: This dataset consists of 32 real-valued features and two classes: 'Normal' and 'Defective.' The features represent various attributes of batteries manufactured by an unspecified company. Examples of such attributes include batch number, internal resistance, and nominal voltage. The primary objective of this dataset is to classify batteries as either 'Normal' or 'Defective.' Each row corresponds to a single battery, with the last column indicating the class label. As with the CKD dataset, some features are normalized, while others are not.

### Your Tasks
You need to download the corresponding data files discussed in Section Introduction and load each dataset into numpy objects (arrays or matrices) in Python. 

Perform some statistical analysis on the datasets e.g. what are the distribution of the two classes? what are the distribution of some of the features? etc. You may visualize your results using histogram plots.
Implement the linear classifiers logistic regression from scratch, by following the equations discussed in class lectures and apply your implemented algorithms to the datasets. To clarify, for example, basic functions such as transpose, shuffle and panda's .mean() and .sum() are okay to be used, but using the train_test_split from sklearn is not ok.

You are free to implement the method in anyway you want, however we recommend to implement the model as python classes (use of constructor is recommended). Each of your model class should have at least two functions: fit and predict. The function fit takes the training data X and its corresponding labels vector y as well as other hyperparameters (such as learning rate) as input, and execute the model training through modifying the model parameters (i.e. W). predict takes a set of test data as input and outputs predicted labels for the input points. Note that you need to convert probabilities to binary 0-1 predictions by thresholding the output at 0.5. (if needed, the ground-truth labels should also be converted to binary 0-1).

Define a function Accu-eval to evaluate the models' accuracy. Accu-eval takes the predicted labels and the true labels as input and outputs the accuracy score.

Implement k-fold cross validation from scratch as a Python class. Use 10-fold cross validation to estimate performance in all of your experiments and you should evaluate performance using accuracy. For model selection, if you have T different models, you should run 10-fold cross validation T times and compare the results. 

At least, complete the followings: test different learning rates for your logistic regression, discuss the run time and accuracy of your logistic regression on both datasets, explore if the accuracy can be improved by a subset of features and/or by inserting new features to the dataset.

This work is part of an ongoing project to explore the applicability of machine learning to predict deliberate self-harm

There are two key variables, X6n and Dsh. X6n is the target variable which the data is modelled on. "X6n" stands for the diagnostic label given to patients with 
a recorded case of deliberate self-harm. Dsh on the other hand is a constructed variable that included all X6ns in addition to others deemed likely to be self-harmers without
having the diagnostic label. The intention of this project is twofold. On the one hand, it is intended to explore the applicability of ML to capture X6n, and on the other, 
it is intended to examine how good the model is at observing Dsh using X6n as the target variable. For these reasons, recall is the primary performance metric. Additionally, 
false positives are not penalized in this project as these (hopefully) are Dshs.

The first script "Data preparation" applies some standard data management; recoding variables, excluding missing cases, and preparing the final dataset.
The second script "Nested CV" runs the hyperparameter optimization, using a gridsearch and cross-validation.
And the third script "XGBoost" initiates the model, and checks the results against the left-out test dataset and the variable Dsh

Because the data is heavily imbalanced - X6n making up 1% of cases - the model XGBoost was chosen. 

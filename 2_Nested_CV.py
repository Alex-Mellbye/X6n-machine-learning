# The website where I found the original code for this script
# https://github.com/pavopax/gists/tree/master/xgboost-with-nested-cv


from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


y = df3['X6n_yes']
X = df3.drop('X6n_yes', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)     # I apply "stratify=y" because the dataset is very imbalanced 


outer_cv = 10
inner_cv = 3
inner_metric = 'recall'
outer_metrics = 'recall'


random_state = 99
n_jobs_1 = 2
n_jobs_2 = 3


params = dict(xgbclassifier__scale_pos_weight=[1, 8, 10, 25, 50, 75, 99, 100],
              xgbclassifier__n_estimators=[10, 50, 100, 500],
              xgbclassifier__max_depth=[10, 50, 100, 500])


clf = make_pipeline(
    StandardScaler(),
    XGBClassifier(random_state=random_state, n_jobs=n_jobs_1))



gcv = GridSearchCV(clf, params, scoring=inner_metric, cv=inner_cv, iid=False,
                   n_jobs=1, return_train_score=False)



results = cross_validate(gcv, X_train, y_train, scoring=outer_metrics, cv=outer_cv,
                         n_jobs=n_jobs_2, return_train_score=False)



print(pd.DataFrame(results))


# The best_params_ line gives me the best hyperparameters for the model's recall metric

gcv.fit(X_train, y_train)
gcv.best_params_

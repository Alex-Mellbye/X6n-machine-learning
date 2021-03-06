# In this script I am initializing the model and comparing the results against a left-out test dataset


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean


############################################ Applying tra_test_split and then initiating the model ##############################


y = df3['X6n_yes']
X = df3.drop('X6n_yes', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)     # I apply "stratify=y" because the dataset is very imbalanced 


xgboost = XGBClassifier(random_state=0, scale_pos_weight=99, n_estimators=10, max_depth=3)
xgboost.fit(X_train, y_train)


############################################ Checking the model against the training dataset ##################################

y_pred = xgboost.predict(X_train)

print("AUC: {}".format(roc_auc_score(y_train, y_pred)))
print("Precision:", precision_score(y_train, y_pred))
print("Recall:",recall_score(y_train, y_pred))
print("F1 score:",f1_score(y_train, y_pred))

print(confusion_matrix(y_train, y_pred))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)              
scores = cross_validate(xgboost, X_train, y_train, cv=skf, scoring="recall" )   

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


############################################ Checking the model against the left-out test dataset  ##############################################

y_pred = xgboost.predict(X_test)
y_proba = xgboost.predict_proba(X_test)[:,1]

print("AUC: {}".format(roc_auc_score(y_test, y_pred)))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("F1 score:",f1_score(y_test, y_pred)) # Resultatet p?? F1 er nesten identisk med resultatet fra kryss-validering (0,52)

print(confusion_matrix(y_test, y_pred)) 
print(sum(y_pred))


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)              
scores = cross_validate(xgboost, X_train, y_train, cv=skf, scoring="recall" )   

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


importances = pd.DataFrame({'feature':X_test.columns,'importance':np.round(xgboost.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.head(15).plot.bar()





############################################ Adding the Dsh variable by matching on index variables  ######################


df5 = pd.concat([X_test, y_test], axis=1)
df_final = pd.merge(df5, df6, left_on="index_var", right_on="index_var2")


preds = pd.Series(y_pred, name="predicts")
predictions = pd.DataFrame(preds)
df_final["predictions"] = predictions["predicts"]


    # Comparing model predictions to the variables Dsh and X6n
print(pd.crosstab(df_final["predictions"], df_final["dsh_yes"]))  
print(pd.crosstab(df_final["X6n_yes"], df_final["dsh_yes"]))     




############################################ Here I produce a recall/precision curve for the test dataset ####################


y_scores = xgboost.predict_proba(X_test)
y_scores = y_scores[:,1]


precision, recall, threshold = precision_recall_curve(y_test, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])


plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()








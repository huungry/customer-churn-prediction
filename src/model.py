from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from preprocess import preprocess_data


#Load training and testing data
X_train, X_test, y_train, y_test = preprocess_data()


#Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

LR_param = {'C': [0.001, 0.01, 0.1, 1, 2, 10], 
            'max_iter': [100, 300, 500, 1000, 1300], 
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs', 'liblinear']}

RF_param = {'n_estimators' : [100, 200, 300, 500, 700], 
            'max_depth' : [None, 1, 5, 10, 15, 20], 
            'min_samples_split' : [2, 4, 5, 8, 10, 15], 
            'min_samples_leaf' : [2, 4, 5, 8, 10, 15], 
            'class_weight' : [None, 'balanced'], 
            'max_features' : ['sqrt', 'log2']}

XGB_param = {'n_estimators' : [100, 200, 300, 400, 500], 
            'max_depth' : [1, 3, 5, 9, 11, 15, 20],
            'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
            'subsample' : [0.7, 0.8, 1.0],
            'colsample_bytree' : [0.7, 0.8, 1.0], 
            'scale_pos_weight' : [1, 2, 5, 7, 9]
            }

#Search for best tuning model performance in each model by cross validation and evaluated by f1 score
LR_search = RandomizedSearchCV(models.get('Logistic Regression'), param_distributions=LR_param, cv=5, scoring='f1', random_state=42, n_jobs=-1)
RF_search = RandomizedSearchCV(models.get('Random Forest'), param_distributions=RF_param, cv=5, scoring='f1', random_state=42, n_jobs=-1)
XGB_search = RandomizedSearchCV(models.get('XGBoost'), param_distributions=XGB_param, cv=5, scoring='f1', random_state=42, n_jobs=-1)

#Fit models
LR_search.fit(X_train, y_train)
RF_search.fit(X_train, y_train)
XGB_search.fit(X_train, y_train)

#Get best version in each model
LR_best = LR_search.best_estimator_
RF_best = RF_search.best_estimator_
XGB_best = XGB_search.best_estimator_


print(classification_report(y_test, LR_best.predict(X_test)))
print(classification_report(y_test, RF_best.predict(X_test)))
print(classification_report(y_test, XGB_best.predict(X_test)))

#Train models and predict
#for name, model in models.items():
#    if name == 'Logistic Regression':
#        lr = model.fit(X_train, y_train)
#        lr_pred = lr.predict(X_test)
#    elif name == 'Random Forest':
#        rf = model.fit(X_train, y_train)
#        rf_pred = rf.predict(X_test)
#    else:
#        xgb = model.fit(X_train, y_train)
#        xgb_pred = xgb.predict(X_test)




#lr_accurate = classification_report(y_test, lr_pred)
#rf_accurate = classification_report(y_test, rf_pred)
#xgb_accurate = classification_report(y_test, xgb_pred)


#print(lr_accurate)
#print(rf_accurate)
#print(xgb_accurate)



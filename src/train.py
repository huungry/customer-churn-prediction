from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from preprocess import preprocess_data
import joblib


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
#XGB has the largest grid, so give the searches enough samples for a fair comparison
LR_search = RandomizedSearchCV(models.get('Logistic Regression'), param_distributions=LR_param, cv=5, scoring='f1', random_state=42, n_jobs=-1, n_iter=20)
RF_search = RandomizedSearchCV(models.get('Random Forest'), param_distributions=RF_param, cv=5, scoring='f1', random_state=42, n_jobs=-1, n_iter=25)
XGB_search = RandomizedSearchCV(models.get('XGBoost'), param_distributions=XGB_param, cv=5, scoring='f1', random_state=42, n_jobs=-1, n_iter=25)

#Fit models
LR_search.fit(X_train, y_train)
RF_search.fit(X_train, y_train)
XGB_search.fit(X_train, y_train)

#Compare and find the best model based on cross validation f1 score
search = {'LR' : LR_search, 'RF' : RF_search, 'XGB' : XGB_search}
best_name = max(search, key=lambda name: search[name].best_score_)
best_model = search.get(best_name).best_estimator_

#Evaluate all three models the same way: CV best f1, test ROC-AUC, and full report
for name, m in search.items():
    model = m.best_estimator_
    prob = model.predict_proba(X_test)[:, 1]    
    print(f"\n===== {name} =====")
    print("CV best f1  : ", round(m.best_score_, 4))
    print("Test ROC-AUC : ", round(roc_auc_score(y_test, prob), 4))
    print('Best parameters : ', m.best_params_)
    print(classification_report(y_test, model.predict(X_test), digits=4))


#Save the best model which is XGBoost for use in predictions
joblib.dump(best_model, 'model/model.joblib')


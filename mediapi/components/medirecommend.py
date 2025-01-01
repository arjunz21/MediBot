import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    mean_squared_error as mse, mean_absolute_error as mae,
    mean_absolute_percentage_error as mape, r2_score, silhouette_score )
# Machine Learning Libraries -- Classification / Regression Libraries
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier

# Clustering Libraries
import scipy.cluster.hierarchy as shc
from sklearn.impute import KNNImputer
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from utils import logging
from utils.helpers import Helpers

class ModelRecommenderConfig:
    dataPath: str = os.path.join(Helpers.basePath + '\\resources', 'data\\')
    modelPath: str = os.path.join('artifacts', 'model.pkl')

class ModelRecommender:
    def __init__(self, X_tr, y_tr, X_te, y_te, progsPath):
        self.modelRecommenderConfig = ModelRecommenderConfig()
        self.X_tr = Helpers.load_object(X_tr)
        self.y_tr = Helpers.load_object(y_tr)
        self.X_te = Helpers.load_object(X_te)
        self.y_te = Helpers.load_object(y_te)
        self.progs = Helpers.load_object(progsPath)
        self.models = {
            'RandomForest': {
                'cls': RandomForestClassifier(random_state=42),
                'params': {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }
            },
            'DecisionTree': {
                'cls': DecisionTreeClassifier(),
                'params': {
                    'criterion':['entropy', 'log_loss', 'gini'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                }
            },
            'KNN': {
                'cls': KNeighborsClassifier(),
                'params': {
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                }
            },
            'SVC': {
                'cls': SVC(kernel='linear'),
                'params': { }
            },
            'LogisticRegression': {
                'cls': LogisticRegression(),
                'params': { }
            },
            'GradientBoostingClassifier': {
                'cls': GradientBoostingClassifier(random_state=42),
                'params': { }
            },
            'MultinomialNB': {
                'cls': MultinomialNB(),
                'params': { }
            }
            # 'XGBoost': XGBClassifier(),
            # 'LGBM': LGBMClassifier(verbose=-1),
        }

    def start(self):
        for name, model in self.models.items():
            print("Model: ", name)
            self.models[name]['name'] = name
            cls = model['cls']
            params = model['params']
            
            # Train Model
            kf = KFold(3, shuffle=True, random_state=42)
            gs = GridSearchCV(estimator=cls, param_grid=params, cv=kf)
            gs.fit(self.X_tr, self.y_tr)
            
            cls.set_params(**gs.best_params_)
            cls.fit(self.X_tr, self.y_tr)
            # cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy', n_jobs=-1)

            # Test Model
            preds = cls.predict(self.X_te)

            # Calculate Accuracy
            accu = accuracy_score(self.y_te, preds)

            # Calculate Confusion Matrix
            cm = confusion_matrix(self.y_te, preds)

            self.models[name]['best_params'] = gs.best_params_
            self.models[name]['metrics'] = accu

            logging.info(str(cm))
            # plt.figure(figsize=(20, 10))
            # plt.title(f'Confusion Matrix for {name} with Accuracy: {accu}')
            # plt.xlabel('Predicted values')
            # plt.ylabel('Actual values')
            # sns.heatmap(data=cm, annot=True, fmt='g', cmap="Blues", xticklabels=progs, yticklabels=progs)
            # plt.show()

        best_model = pd.DataFrame.from_dict(self.models, orient='index').reset_index().sort_values(by=['metrics'], ascending=[False]).iloc[0]
        print(best_model)
        logging.info(f"Best Found: {str(best_model)}")
        Helpers.save_object(self.modelRecommenderConfig.modelPath, best_model['cls'])
        logging.info("Best Model Saved as Pickle file")

        return self.modelRecommenderConfig.modelPath, best_model

    def trainModel(self):
        pass

    def metrics(self):
        pass

    def validate(self):
        pass

    def predict(self):
        model = Helpers.load_object(self.modelRecommenderConfig.modelPath)
        model.fit(self.X_tr, self.y_tr)
        txt = "itching coughing sleeping aching".split()
        inp = np.zeros(len(self.X_te.columns))
        for s in txt:
            if s in self.X_te.columns:
                inp[list(self.X_te.columns).index(s)] = 1
        predDisease = model.predict(inp.reshape(1, -1))
        print(predDisease, self.progs[predDisease[0]])
        return self.progs[predDisease[0]]
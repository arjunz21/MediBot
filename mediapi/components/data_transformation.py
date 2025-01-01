import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
# from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel

# Transformation Libraries
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

from utils import logging
from utils.helpers import Helpers

@dataclass
class DataTransformationConfig:
    dataPath: str = os.path.join(Helpers.basePath + '\\resources', 'data\\')
    X_trPath: str = os.path.join('artifacts', 'X_tr')
    y_trPath: str = os.path.join('artifacts', 'y_tr')
    X_tePath: str = os.path.join('artifacts', 'X_te')
    y_tePath: str = os.path.join('artifacts', 'y_te')

class DataTransformation:
    def __init__(self, trdfPath, progsPath):
        self.dataTransformationConfig = DataTransformationConfig()
        self.sym_trdf = Helpers.load_object(trdfPath)
        self.progs = Helpers.load_object(progsPath)

    def start(self):
        logging.info("Data Transformation started.")
        self.X = self.sym_trdf.drop(columns=['prognosis'])
        self.y = self.sym_trdf['prognosis']
        # logging.info(str(self.X.shape, self.y.shape))

        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        # logging.info(str(self.X_tr.shape, self.X_te.shape, self.y_tr.shape, self.y_te.shape))

        le = LabelEncoder()
        le.fit_transform(self.progs)
        self.y_tr = le.transform(self.y_tr)
        self.y_te = le.transform(self.y_te)
        logging.info(self.X_te.columns)

        Helpers.save_object(self.dataTransformationConfig.X_trPath, self.X_tr)
        Helpers.save_object(self.dataTransformationConfig.y_trPath, self.y_tr)
        Helpers.save_object(self.dataTransformationConfig.X_tePath, self.X_te)
        Helpers.save_object(self.dataTransformationConfig.y_tePath, self.y_te)
        
        return self.dataTransformationConfig.X_trPath, self.dataTransformationConfig.y_trPath, self.dataTransformationConfig.X_tePath, self.dataTransformationConfig.y_tePath

        
        
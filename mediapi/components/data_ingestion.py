import os
# import sys
import pandas as pd
from dataclasses import dataclass

from utils import logging
from utils.helpers import Helpers

@dataclass
class DataIngestionConfig:
    dataPath: str = os.path.join(Helpers.basePath + '\\resources', 'data\\')
    trainPath: str = os.path.join('artifacts', 'train.csv')
    symsPath: str = os.path.join('artifacts', 'syms.csv')
    severityPath: str = os.path.join('artifacts', 'severity.csv')
    progsPath: str = os.path.join('artifacts', 'progs')
    symtnPath: str = os.path.join('artifacts', 'symtn')

class DataIngestion:
    def __init__(self):
        self.dataIngestionConfig = DataIngestionConfig()

    def start(self):
        try:
            logging.info("Data Ingestion started. Reading Data")
            # base = "C:\\ArjunData\\AppDev\\python\\MediBot\\resources\\data\\"
            self.sym_trdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'training.csv')
            self.sym_trdf['prognosis'] = self.sym_trdf['prognosis'].str.lower()
            sym_descdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'description.csv')
            sym_severdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'symptom-severity.csv').rename(columns={'Symptom':'Disease'})
            sym_severdf = sym_severdf.astype({'weight':'object'})
            symptomsdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'symtoms.csv').drop(columns=['Unnamed: 0'])
            symptomsdf = symptomsdf.groupby('Disease').apply(Helpers.symtonsList)
            symptomsdf = pd.DataFrame(symptomsdf, columns=['Symtoms']).reset_index()

            dietsdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'diets.csv')
            medsdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'medications.csv')
            precautionsdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'precautions.csv')
            precautionsdf['Precautions'] = precautionsdf[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
            precautionsdf.drop(columns=['Unnamed: 0','Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], inplace=True)
            precautionsdf['Disease'] = precautionsdf['Disease'].str.lower()
            workoutdf = pd.read_csv(self.dataIngestionConfig.dataPath + 'workout.csv').drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
            workoutdf = workoutdf.groupby('disease', as_index=False).agg(list)
            workoutdf['Disease'] = workoutdf['disease'].str.lower()
            workoutdf.drop(columns=['disease'], inplace=True)
            
            logging.info("Merging symtoms related data")
            self.symsdf = pd.merge(symptomsdf, sym_descdf, how='left', on='Disease')
            # self.symsdf = pd.merge(self.symsdf, sym_severdf, how='inner', on='Disease')
            self.symsdf = pd.merge(self.symsdf, medsdf, how='left', on='Disease')
            self.symsdf = pd.merge(self.symsdf, dietsdf, how='left', on='Disease')
            self.symsdf['Disease'] = self.symsdf['Disease'].str.lower()
            self.symsdf = pd.merge(self.symsdf, precautionsdf, how='left', on='Disease')
            self.symsdf = pd.merge(self.symsdf, workoutdf, how='left', on='Disease').dropna()
            progs = self.sym_trdf['prognosis'].unique()
            symtn = list(set([item.strip() for sublist in self.symsdf['Symtoms'] for item in sublist if type(item) != float]))
            
            Helpers.save_object(self.dataIngestionConfig.trainPath, self.sym_trdf)
            Helpers.save_object(self.dataIngestionConfig.symsPath, self.symsdf)
            Helpers.save_object(self.dataIngestionConfig.severityPath, sym_severdf)
            Helpers.save_object(self.dataIngestionConfig.progsPath, progs)
            Helpers.save_object(self.dataIngestionConfig.symtnPath, symtn)
            logging.info("Data Ingestion completed.")
        except Exception as e:
            logging.error("Error in DataIngestion:" + str(e))

        return self.dataIngestionConfig.trainPath, self.dataIngestionConfig.progsPath, self.dataIngestionConfig.symsPath, self.dataIngestionConfig.symtnPath, self.dataIngestionConfig.severityPath
    
    def info(self):
        logging.info("Checking Information about Dataset")
        print(self.symsdf.info())
        print(self.symsdf.describe())
        print(self.symsdf.head())
        print(self.sym_trdf.info())
        print(self.sym_trdf.describe().T)
        print(self.sym_trdf.head())

    def visuals(self):
        logging.info("Visualizing the Data and storing into images.")
        # wordCloudBar(sym_trdf['prognosis'])

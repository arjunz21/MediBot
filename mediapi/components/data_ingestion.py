import os
# import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from utils import logging

@dataclass
class DataIngestionConfig:
    curPath: str = os.getcwd()
    rawDataPath: str = os.path.join('artifacts', 'raw.csv')
    trDataPath: str = os.path.join('artifacts', 'train.csv')
    teDataPath: str = os.path.join('artifacts', 'test.csv')
    valDataPath: str = os.path.join('artifacts', 'val.csv')


class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    

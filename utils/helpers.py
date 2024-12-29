import os
import pickle
import configparser
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

from utils import logging

config = configparser.RawConfigParser()
config.read(os.path.join('resources', 'config.ini'))


class Helpers:
    basePath = os.getcwd()

    @staticmethod
    def symtonsList(grp):
        syms = grp[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']].values.tolist()
        return list(set([str(item).strip() for sublist in syms for item in sublist]))

    @staticmethod
    def wordCloudBar(sr):
        # Create wordCLoud
        cnt = Counter([str(item).strip() for sublist in sr for item in sublist])
        print(cnt.most_common(5))
        w = WordCloud(max_words = 5000 , width = 1600 , height = 800).generate_from_frequencies(frequencies=dict(cnt))
        plt.figure(figsize=(10, 5))
        plt.axis('off')
        plt.imshow(w)
    
    @staticmethod
    def save_object(filePath, obj):
        try:
            logging.info(f"Saving Object: {filePath}")
            dirPath = os.path.dirname(filePath)
            os.makedirs(dirPath, exist_ok=True)
            with open(filePath, "wb") as fileObj:
                pickle.dump(obj, fileObj)
        
        except Exception as e:
            logging.info("Error occured in saving object: ", e)
    
    @staticmethod
    def load_object(filePath):
        try:
            logging.info("Loading the object")
            with open(filePath, "rb") as fileObj:
                obj = pickle.load(fileObj)
            return obj
        
        except Exception as e:
            logging.info("Error occured in loading object: ", e)
    
    @staticmethod
    def read_config(section, key):
        return config.get(section, key, fallback=None)
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
#import dill


def save_object(filepath,object):
    try:
        dir_path = os.path.dirname(filepath)

        logging.info("creating directory for saving object")
        os.makedirs(dir_path,exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(object,f)
        

    except Exception as e:
        logging.info("Error occurred while saving object")
        raise CustomException(e,sys)

def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        logging.info("model evaluation started")
        model_report = {}
        trained_models = []
        for i in range(len(models)):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            trained_models.append(model)

            test_model_score = r2_score(y_test,y_test_pred)

            model_report[list(models.keys())[i]]= test_model_score

            logging.info(f"{list(models.keys())[i]}= {test_model_score}")
        logging.info("model evaluation finished")
        return (model_report,trained_models)
    except Exception as e:
        logging.error(e)
        logging.info("Error occured while evaluating models")
        raise CustomException(e,sys)
    
def load_object(filepath):
    try:
        with open(filepath,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)

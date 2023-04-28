import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.utils import save_object,load_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Starting model training')
            logging.info('splitting the transformed arrays into dependent and independent arrays')
            X_train = train_arr[:,0:-1]
            y_train = train_arr[:,-1]

            X_test = test_arr[:,0:-1]
            y_test = test_arr[:,-1]

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elastic Net' : ElasticNet()
            }

            model_report,trained_models = evaluate_model(X_train = X_train,X_test = X_test, y_train= y_train,y_test =  y_test,models = models)

            print(model_report)
            print(trained_models)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            logging.info("model report generated")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            ####################################################################################################
            # best_model1 = trained_models[list(model_report.values()).index(best_model_score)]
            
            # yar = best_model1.predict(X_test)

            # print(f"this is trial score {r2_score(y_test,yar)}")
            ####################################################################################################
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 filepath=self.model_trainer_config.trained_model_file_path,
                 object=best_model
            )




        except Exception as e:
            raise CustomException(e,sys)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Started data ingestion")

            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset read")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)

            logging.info("raw data file created successfully")

            logging.info("Initializing train_test_split")
            train_df,test_df = train_test_split(df,test_size=0.3,random_state=42)
            logging.info("df splitted")

            train_df.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_path,index = False,header=True)

            logging.info("Data Ingestion Completed")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )



            
        except Exception as e:
            logging.info("Something went wrong during data ingestion")
            logging.error(e)
            raise CustomException(e,sys)



        
if __name__ == "__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)

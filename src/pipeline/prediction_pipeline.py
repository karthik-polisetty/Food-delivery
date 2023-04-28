import sys
import os
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):

        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            scaled_df = preprocessor.transform(features)
            pred_value = model.predict(scaled_df)
            return pred_value
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,carat:int,cut:str,color:str,
                 clarity:str,depth:int,table:int,
                 x:int,y:int,z:int):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            custom_input_data_dict = {
                'carat':[self.carat],
                'cut': [self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z]
            }

            return pd.DataFrame(custom_input_data_dict)
        except Exception as e:
            raise CustomException(e,sys)

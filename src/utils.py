import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from geopy.distance import geodesic
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
            model_name = list(models.keys())[i]

            model.fit(X_train, y_train)

            #y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            trained_models.append(model)

            test_model_score = r2_score(y_test,y_test_pred)

            model_report[model_name]= test_model_score

            logging.info(f"{model_name}= {test_model_score}")
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
    

def distcalculate(df):

    try:
        for i in range(len(df)):
            df.loc[i, 'distance'] = geodesic((df.loc[i, 'Restaurant_latitude'], 
                                                df.loc[i, 'Restaurant_longitude']),
                                                (df.loc[i, 'Delivery_location_latitude'], 
                                                df.loc[i, 'Delivery_location_longitude'])).km
    except Exception as e:
        raise CustomException(e,sys)
    
def time_format(i):
    try:
        if len(i)>=5:
            if i[2]!= ":":
                return np.nan
            else:
                return i
        else:
            return np.nan 
    except Exception as e:
        raise CustomException(e,sys)
    
def preprocess_dataset(df):
    try:

        # Claculating the distance between hotel and delivey location using disctcalculate function 
        distcalculate(df)

        # Dropping all the 4 columns as new distance column has been created                             
        df = df.drop(labels=['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude'],axis=1)
        
        
        
        # Some junk values in time columns are replaced with null values
        df['Time_Orderd']=df['Time_Orderd'].astype(str).apply(time_format)
        df['Time_Order_picked']=df['Time_Order_picked'].astype(str).apply(time_format)
        
        
        # Splitting the time into mins and hours to find difference between them
        df['Ordered_Hour']=df['Time_Orderd'].str.split(':').str[0]
        df['Ordered_Min']=df['Time_Orderd'].str.split(':').str[1]
        
        df['Picked_Hour']=df['Time_Order_picked'].str.split(':').str[0]
        df['Picked_Min']=df['Time_Order_picked'].str.split(':').str[1]
        
        # Dropping the time columns after dividing and forming new columns from that
        df.drop(labels=['Time_Orderd','Time_Order_picked'],inplace=True,axis=1)
        df.drop(labels=['ID','Delivery_person_ID'],inplace=True,axis=1)
        
        # calculating the time difference 
        df['time_diff']=(df['Picked_Hour'].astype(float)*60+df['Picked_Min'].astype(float))-(df['Ordered_Hour'].astype(float)*60+df['Ordered_Min'].astype(float))
        
        # ddropping these columns after calculating the time difference
        df.drop(labels=['Ordered_Hour','Ordered_Min','Picked_Hour','Picked_Min'],inplace=True,axis=1)
        
        # dropping the order date column
        df.drop('Order_Date',inplace=True,axis=1)
        
        return df
    except Exception as e:
        raise CustomException(e,sys)
    


        
        
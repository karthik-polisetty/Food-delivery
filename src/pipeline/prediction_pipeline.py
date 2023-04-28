import sys
import os
import pandas as pd
import numpy as np
from src.utils import load_object,preprocess_dataset
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

            data = preprocess_dataset(features)
            scaled_df = preprocessor.transform(data)
            pred_value = model.predict(scaled_df)
            return pred_value
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self, ID, Delivery_person_ID, Delivery_person_Age,
                 Delivery_person_Ratings, Restaurant_latitude,
                 Restaurant_longitude, Delivery_location_latitude,
                 Delivery_location_longitude, Order_Date, Time_Orderd,
                 Time_Order_picked, Weather_conditions, Road_traffic_density,
                 Vehicle_condition, Type_of_order, Type_of_vehicle,
                 multiple_deliveries, Festival, City):
        self.ID = ID
        self.Delivery_person_ID = Delivery_person_ID
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Order_Date = Order_Date
        self.Time_Orderd = Time_Orderd
        self.Time_Order_picked = Time_Order_picked
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_input_data_dict = {
                'ID': [self.ID],
                'Delivery_person_ID': [self.Delivery_person_ID],
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Restaurant_latitude': [self.Restaurant_latitude],
                'Restaurant_longitude': [self.Restaurant_longitude],
                'Delivery_location_latitude': [self.Delivery_location_latitude],
                'Delivery_location_longitude': [self.Delivery_location_longitude],
                'Order_Date': [self.Order_Date],
                'Time_Orderd': [self.Time_Orderd],
                'Time_Order_picked': [self.Time_Order_picked],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Vehicle_condition': [self.Vehicle_condition],
                'Type_of_order': [self.Type_of_order],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'multiple_deliveries': [self.multiple_deliveries],
                'Festival': [self.Festival],
                'City': [self.City]
            }

            return pd.DataFrame(custom_input_data_dict)
        except Exception as e:
            raise CustomException(e,sys)

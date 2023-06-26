import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import boto3
from data_processing import load_and_preprocess_data
from model import train_model, evaluate_model
from cloud_services import upload_to_s3

def get_weather(city_name):
    response = requests.get(
        f"http://api.weatherapi.com/v1/current.json?key=YOUR_API_KEY&q={city_name}"
    )
    data = response.json()
    return data["location"]["name"], data["current"]["temp_c"]

upload_to_s3('weather_data.csv', 'my_weather_app_bucket')

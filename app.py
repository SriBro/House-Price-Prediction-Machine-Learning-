import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import subprocess
import webbrowser

st.title("House Price Prediction for California District of USA")
st.header("Enter numeric values")

med_inc = st.number_input("Median Income: ")
house_age = st.number_input("House Age: ")
ave_rooms = st.number_input("Average Rooms: ")
ave_bedrms = st.number_input("Average Bed Rooms: ")
population = st.number_input("Population: ")
ave_occup = st.number_input("Average Occupation: ")
latitude = st.number_input("Latitude: ")
st.write("(Valid range: 32.5<=Latitude<=42)")
longitude = st.number_input("Longitude: ")
st.write("(Valid range: -124.5<=Longitude<=-114.5)")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    model1 = st.button('Predict using Model 1')
with col2:
    model2 = st.button("Predict using Model 2")
with col3:
    model3 = st.button("Predict using Model 3")
with col4:
    model4 = st.button("Predict using Model 4")
with col5:
    model5 = st.button("Predict using Model 5")

with col2:
    compare = st.button('Compare Models')
with col4:
    know = st.button('Know This Dataset')

def display_hello():
    subprocess.run(["python","info.py"])

if model1:
    inputData = []
    invalid = []
    def is_valid_number(input_str):
        # Check if the input string is a valid number
        try:
            float(input_str)
            return True
        except ValueError:
            return False

    def is_valid_latitude(value):
        return 32.5 <= float(value) <= 42.0

    def is_valid_longitude(value):
        return -124.5 <= float(value) <= -114.5

    # Validate inputs and add to inputData if valid
    if is_valid_number(med_inc):
        inputData.append(float(med_inc))
    else:
        invalid.append("Invalid input for Median Income!")

    if is_valid_number(house_age):
        inputData.append(float(house_age))
    else:
        invalid.append("Invalid input for House Age!")

    if is_valid_number(ave_rooms):
        inputData.append(float(ave_rooms))
    else:
        invalid.append("Invalid input for Average Rooms!")

    if is_valid_number(ave_bedrms):
        inputData.append(float(ave_bedrms))
    else:
        invalid.append("Invalid input for Average Bedrooms!")

    if is_valid_number(population):
        inputData.append(float(population))
    else:
        invalid.append("Invalid input for Population!")

    if is_valid_number(ave_occup):
        inputData.append(float(ave_occup))
    else:
        invalid.append("Invalid input for Average Occupation!")

    if is_valid_number(latitude) and is_valid_latitude(latitude):
        inputData.append(float(latitude))
    else:
        invalid.append("Invalid input for Latitude! Valid range is between 32.5N and 42N inclusive.")

    if is_valid_number(longitude) and is_valid_longitude(longitude):
        inputData.append(float(longitude))
    else:
        invalid.append("Invalid input for Longitude! Valid range is between -124.5W and -114.5W inclusive.")

    length = len(invalid)
    for i in range(0, length):
        st.markdown(f'<font color="red" style="font-size: 20px; font-style: italic;">{invalid[i]}</font>', unsafe_allow_html=True)

    if not invalid:
      # Load California housing dataset
      california_housing = fetch_california_housing()
      data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
      target = pd.DataFrame(data=california_housing.target, columns=["target"])

      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

      # Standardize the features
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      #  Build and train a LGBM Regressor model
      # n_estimators=100 specifies the number of boosting rounds
      # random_state=42 sets the seed for reproducibility
      #model = LGBMRegressor(n_estimators=100, random_state=42)
      #model.fit(X_train_scaled, y_train.values.ravel())

      # Dumping the trained model
      save_directory = "D:\\ganesh"
      model_file_path = os.path.join(save_directory, 'lgbmr_regressor_model_updated.joblib')
      #joblib.dump(model,model_file_path)
      loaded_model = joblib.load(model_file_path)

      # Make predictions on the test set
      new_data = pd.DataFrame(data=[inputData], columns=california_housing.feature_names)
      new_data_scaled = scaler.transform(new_data)
      predictions = loaded_model.predict(new_data_scaled)
      predicted_value_dollar = predictions[0]
      predicted_value_dollar*=100000
      predicted_value_rupee = predicted_value_dollar*82.74
      st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is    {predicted_value_dollar} in dollars</font>', unsafe_allow_html=True)
      st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is   {predicted_value_rupee} in rupees</font>', unsafe_allow_html=True)



if model2:
    inputData = []
    invalid = []
    
    def is_valid_number(input_str):
        # Check if the input string is a valid number
        try:
            float(input_str)
            return True
        except ValueError:
            return False

    def is_valid_latitude(value):
        return 32.5 <= float(value) <= 42.0

    def is_valid_longitude(value):
        return -124.5 <= float(value) <= -114.5

    # Validate inputs and add to inputData if valid
    if is_valid_number(med_inc):
        inputData.append(float(med_inc))
    else:
        invalid.append("Invalid input for Median Income!")

    if is_valid_number(house_age):
        inputData.append(float(house_age))
    else:
        invalid.append("Invalid input for House Age!")

    if is_valid_number(ave_rooms):
        inputData.append(float(ave_rooms))
    else:
        invalid.append("Invalid input for Average Rooms!")

    if is_valid_number(ave_bedrms):
        inputData.append(float(ave_bedrms))
    else:
        invalid.append("Invalid input for Average Bedrooms!")

    if is_valid_number(population):
        inputData.append(float(population))
    else:
        invalid.append("Invalid input for Population!")

    if is_valid_number(ave_occup):
        inputData.append(float(ave_occup))
    else:
        invalid.append("Invalid input for Average Occupation!")

    if is_valid_number(latitude) and is_valid_latitude(latitude):
        inputData.append(float(latitude))
    else:
        invalid.append("Invalid input for Latitude! Valid range is between 32.5N and 42N inclusive.")

    if is_valid_number(longitude) and is_valid_longitude(longitude):
        inputData.append(float(longitude))
    else:
        invalid.append("Invalid input for Longitude! Valid range is between -124.5W and -114.5W inclusive.")

    length = len(invalid)

    for i in range(0, length):
        st.markdown(f'<font color="red" style="font-size: 20px; font-style: italic;">{invalid[i]}</font>', unsafe_allow_html=True)
    

    # Make predictions only if all inputs are valid
    if not invalid:
        # Load California housing dataset
        california_housing = fetch_california_housing()
        data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
        target = pd.DataFrame(data=california_housing.target, columns=["target"])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and train the Random Forest Regressor model
        #model = XGBRegressor(n_estimators=100, random_state=42)
        #model.fit(X_train_scaled, y_train.values.ravel())

        # Dumping the trained model
        save_directory = "D:\\ganesh"
        model_file_path = os.path.join(save_directory, 'XGB_regressor_model.joblib')
        #joblib.dump(model,model_file_path)
        loaded_model = joblib.load(model_file_path)

        # Make predictions on the test set
        new_data = pd.DataFrame(data=[inputData], columns=california_housing.feature_names)
        new_data_scaled = scaler.transform(new_data)
        predictions = loaded_model.predict(new_data_scaled)
        predicted_value_dollar = predictions[0]
        predicted_value_dollar*=100000
        predicted_value_rupee = predicted_value_dollar*82.74
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is    {predicted_value_dollar} in dollars</font>', unsafe_allow_html=True)
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is   {predicted_value_rupee} in rupees</font>', unsafe_allow_html=True)


if model3:
    inputData = []
    invalid = []
    
    def is_valid_number(input_str):
        # Check if the input string is a valid number
        try:
            float(input_str)
            return True
        except ValueError:
            return False

    def is_valid_latitude(value):
        return 32.5 <= float(value) <= 42.0

    def is_valid_longitude(value):
        return -124.5 <= float(value) <= -114.5

    # Validate inputs and add to inputData if valid
    if is_valid_number(med_inc):
        inputData.append(float(med_inc))
    else:
        invalid.append("Invalid input for Median Income!")

    if is_valid_number(house_age):
        inputData.append(float(house_age))
    else:
        invalid.append("Invalid input for House Age!")

    if is_valid_number(ave_rooms):
        inputData.append(float(ave_rooms))
    else:
        invalid.append("Invalid input for Average Rooms!")

    if is_valid_number(ave_bedrms):
        inputData.append(float(ave_bedrms))
    else:
        invalid.append("Invalid input for Average Bedrooms!")

    if is_valid_number(population):
        inputData.append(float(population))
    else:
        invalid.append("Invalid input for Population!")

    if is_valid_number(ave_occup):
        inputData.append(float(ave_occup))
    else:
        invalid.append("Invalid input for Average Occupation!")

    if is_valid_number(latitude) and is_valid_latitude(latitude):
        inputData.append(float(latitude))
    else:
        invalid.append("Invalid input for Latitude! Valid range is between 32.5N and 42N inclusive.")

    if is_valid_number(longitude) and is_valid_longitude(longitude):
        inputData.append(float(longitude))
    else:
        invalid.append("Invalid input for Longitude! Valid range is between -124.5W and -114.5W inclusive.")

    length = len(invalid)
    for i in range(0, length):
        st.markdown(f'<font color="red" style="font-size: 20px; font-style: italic;">{invalid[i]}</font>', unsafe_allow_html=True)
    

    # Make predictions only if all inputs are valid
    if not invalid:
        # Load California housing dataset
        california_housing = fetch_california_housing()
        data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
        target = pd.DataFrame(data=california_housing.target, columns=["target"])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and train the Random Forest Regressor model
        # model = RandomForestRegressor(n_estimators=100, random_state=42)
        # model.fit(X_train_scaled, y_train.values.ravel())

        # Dumping the trained model
        save_directory = "D:\\ganesh"
        model_file_path = os.path.join(save_directory, 'random_forest_model.joblib')
        # joblib.dump(model,model_file_path)
        loaded_model = joblib.load(model_file_path)

        # Make predictions on the test set
        new_data = pd.DataFrame(data=[inputData], columns=california_housing.feature_names)
        new_data_scaled = scaler.transform(new_data)
        predictions = loaded_model.predict(new_data_scaled)
        predicted_value_dollar = predictions[0]
        predicted_value_dollar*=100000
        predicted_value_rupee = predicted_value_dollar*82.74
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is    {predicted_value_dollar} in dollars</font>', unsafe_allow_html=True)
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is   {predicted_value_rupee} in rupees</font>', unsafe_allow_html=True)


if model4:
    inputData = []
    invalid = []

    def is_valid_number(input_str):
        # Check if the input string is a valid number
        try:
            float(input_str)
            return True
        except ValueError:
            return False

    def is_valid_latitude(value):
        return 32.5 <= float(value) <= 42.0

    def is_valid_longitude(value):
        return -124.5 <= float(value) <= -114.5
    
    # Validate inputs and add to inputData if valid
    if is_valid_number(med_inc):
        inputData.append(float(med_inc))
    else:
        invalid.append("Invalid input for Median Income!")

    if is_valid_number(house_age):
        inputData.append(float(house_age))
    else:
        invalid.append("Invalid input for House Age!")

    if is_valid_number(ave_rooms):
        inputData.append(float(ave_rooms))
    else:
        invalid.append("Invalid input for Average Rooms!")

    if is_valid_number(ave_bedrms):
        inputData.append(float(ave_bedrms))
    else:
        invalid.append("Invalid input for Average Bedrooms!")

    if is_valid_number(population):
        inputData.append(float(population))
    else:
        invalid.append("Invalid input for Population!")

    if is_valid_number(ave_occup):
        inputData.append(float(ave_occup))
    else:
        invalid.append("Invalid input for Average Occupation!")

    if is_valid_number(latitude) and is_valid_latitude(latitude):
        inputData.append(float(latitude))
    else:
        invalid.append("Invalid input for Latitude! Valid range is between 32.5N and 42N inclusive!")

    if is_valid_number(longitude) and is_valid_longitude(longitude):
        inputData.append(float(longitude))
    else:
        invalid.append("Invalid input for Longitude! Valid range is between -124.5W and -114.5W inclusive!")

    length = len(invalid)

    for i in range(0, length):
        st.markdown(f'<font color="red" style="font-size: 20px; font-style: italic;">{invalid[i]}</font>', unsafe_allow_html=True)


    # Make predictions only if all inputs are valid
    if not invalid:
        # Load California housing dataset
        california_housing = fetch_california_housing()
        data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
        target = pd.DataFrame(data=california_housing.target, columns=["target"])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and train the Gradient Boosting Regressor model
        #model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        #model.fit(X_train_scaled, y_train.values.ravel())

        # Dumping the trained model
        save_directory = "D:\\ganesh"
        model_file_path = os.path.join(save_directory, 'gradient_boosting_regressor_model.joblib')
        #joblib.dump(model,model_file_path)
        loaded_model = joblib.load(model_file_path)

        # Make predictions on the test set
        new_data = pd.DataFrame(data=[inputData], columns=california_housing.feature_names)
        new_data_scaled = scaler.transform(new_data)
        predictions = loaded_model.predict(new_data_scaled)
        predicted_value_dollar = predictions[0]
        predicted_value_dollar*=100000
        predicted_value_rupee = predicted_value_dollar*82.74
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is    {predicted_value_dollar} in dollars</font>', unsafe_allow_html=True)
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is   {predicted_value_rupee} in rupees</font>', unsafe_allow_html=True)

if model5:
    inputData = []
    invalid = []
    def is_valid_number(input_str):
        # Check if the input string is a valid number
        try:
            float(input_str)
            return True
        except ValueError:
            return False

    def is_valid_latitude(value):
        return 32.5 <= float(value) <= 42.0

    def is_valid_longitude(value):
        return -124.5 <= float(value) <= -114.5

    
    # Validate inputs and add to inputData if valid
    if is_valid_number(med_inc):
        inputData.append(float(med_inc))
    else:
        invalid.append("Invalid input for Median Income!")

    if is_valid_number(house_age):
        inputData.append(float(house_age))
    else:
        invalid.append("Invalid input for House Age!")

    if is_valid_number(ave_rooms):
        inputData.append(float(ave_rooms))
    else:
        invalid.append("Invalid input for Average Rooms!")

    if is_valid_number(ave_bedrms):
        inputData.append(float(ave_bedrms))
    else:
        invalid.append("Invalid input for Average Bedrooms!")

    if is_valid_number(population):
        inputData.append(float(population))
    else:
        invalid.append("Invalid input for Population!")

    if is_valid_number(ave_occup):
        inputData.append(float(ave_occup))
    else:
        invalid.append("Invalid input for Average Occupation!")

    if is_valid_number(latitude) and is_valid_latitude(latitude):
        inputData.append(float(latitude))
    else:
        invalid.append("Invalid input for Latitude! Valid range is between 32.5N and 42N inclusive.")

    if is_valid_number(longitude) and is_valid_longitude(longitude):
        inputData.append(float(longitude))
    else:
        invalid.append("Invalid input for Longitude! Valid range is between -124.5W and -114.5W inclusive.")

    length = len(invalid)

    for i in range(0, length):
        st.markdown(f'<font color="red" style="font-size: 20px; font-style: italic;">{invalid[i]}</font>', unsafe_allow_html=True)

    # Make predictions only if all inputs are valid
    if not invalid:
        # Load data
        california_housing = fetch_california_housing()
        data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
        target = pd.DataFrame(data=california_housing.target, columns=["target"])

        # Split data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

        # Standardize data for data consistency
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and train the SVR model
        #model = SVR()
        #model.fit(X_train_scaled, y_train.values.ravel())

        # Dumping the trained model
        save_directory = "D:\\ganesh"
        model_file_path = os.path.join(save_directory, 'support_vector_regressor_model.joblib')
        #joblib.dump(model, model_file_path)
        loaded_model = joblib.load(model_file_path)

        # Make predictions on the test set
        new_data = pd.DataFrame(data=[inputData], columns=california_housing.feature_names)
        new_data_scaled = scaler.transform(new_data)
        predictions = loaded_model.predict(new_data_scaled)
        predicted_value_dollar = predictions[0]
        predicted_value_dollar*=100000
        predicted_value_rupee = predicted_value_dollar*82.74
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is    {predicted_value_dollar} in dollars</font>', unsafe_allow_html=True)
        st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">Your Predicted House Price is   {predicted_value_rupee} in rupees</font>', unsafe_allow_html=True)

def compare_model_window():
    models = ['LGBMR', 'XGBR', 'RFR', 'GBR', 'SVR']
    accuracy_values = [78.9, 77.8, 74.5, 70.7, 67.6]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='lightgrey')
    ax.tick_params(axis='y', colors='lightgrey')

    bars = ax.bar(models, accuracy_values, color='purple')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval}%', ha='center', va='bottom', color='black')

    ax.set_xlabel('Models', color='lightgrey')
    ax.set_ylabel('Accuracy', color='lightgrey')

    st.pyplot(fig)

def display_image():
    st.title("Model Accuracy Comparison")
    st.image('E:/image/compare_models.png', caption='5 Regression Models', use_column_width=True)
if compare:
    display_image()

if know:
    st.title("California Dataset")
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">1. This Dataset can be fetched from the function of scikit-learn library called fetch_california_housing</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">2. This model predicts the price of a house of California state(United States) based on some features</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">3. Here the houses are assumed to be grouped and represent a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;">4. Features:</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (i) MedInc: Middle value of all the incomes of every house in a particular California District', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (ii) HouseAge: Middle value of all the ages of every house in a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (iii) AveRooms: Average number of rooms per house in a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (iv) AveBedrms: Average number of bed rooms per house in a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (v) Population: Population of a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (vi) AveOccup: Average number of people residing in a house of a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (vii) Latitude: Latitude(horizontal) co-ordinate of a center point of a particular California District</font>', unsafe_allow_html=True)
    st.markdown(f'<font color="yellow" style="font-size:20px; font-style:italic;"> (viii) Longitude: Longitude(vertical) co-ordinate of a center point of a particular California District</font>', unsafe_allow_html=True)
    
    




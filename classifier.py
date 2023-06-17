# import required libraries
import pandas as pd
import joblib

# load the trained Random Forest model
rf_model = joblib.load("fraud_detection_model2.sav")

# define the list of column names for input data
input_cols = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last',
              'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job',
              'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

# take user input for transaction details
input_data = []
for col in input_cols:
    val = input("Enter {}: ".format(col))
    input_data.append(val)

# create a dataframe from user input data
input_df = pd.DataFrame([input_data], columns=input_cols)

# preprocess the input data
input_df['merchant'] = pd.Categorical(input_df['merchant'], categories=rf_model['merchant'].categories_, ordered=False)
input_df['category'] = pd.Categorical(input_df['category'], categories=rf_model['category'].categories_, ordered=False)
input_df['gender'] = pd.Categorical(input_df['gender'], categories=rf_model['gender'].categories_, ordered=False)
input_df['job'] = pd.Categorical(input_df['job'], categories=rf_model['job'].categories_, ordered=False)

input_categorical = pd.get_dummies(input_df[['merchant', 'category', 'gender', 'job']])

input_numerical = input_df.drop(['merchant', 'category', 'gender', 'job'], axis=1)

# preprocess date column and convert to Unix time
input_numerical['trans_date_trans_time'] = pd.to_datetime(input_numerical['trans_date_trans_time'])
input_numerical['trans_date_trans_time'] = input_numerical['trans_date_trans_time'].apply(lambda x: pd.datetime.timestamp(x))

# ensure the input data has the same columns as the training data
input_data = pd.concat([input_categorical, input_numerical], axis=1)
missing_cols = set(rf_model.feature_names_) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0
input_data = input_data[rf_model.feature_names_]

# make a prediction on the input data
prediction = rf_model.predict(input_data)

# print the prediction result
if prediction[0] == 0:
    print("The transaction is not fraudulent.")
else:
    print("The transaction is fraudulent!")

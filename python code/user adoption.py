from re import X
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load dataset for data exploration and prediction analysis 
user_df = pd.read_csv('/Users/georgetteturkson/Downloads/cleaned_users_logininfo.csv', encoding='latin1')
print(user_df.head())

engagement_df=pd.read_csv('/Users/georgetteturkson/Downloads/cleaned_user_engagement.csv', encoding='latin1')
print("Columns in engagement_df:")
print(engagement_df.head())

#summary statistics 
print(user_df.describe(include='all'))
print(engagement_df.describe(include='all'))

#check and remove spaces from column names
print(user_df.columns)
user_df.columns = user_df.columns.str.strip()
print(user_df.columns)

print(engagement_df.columns)
engagement_df.columns = engagement_df.columns.str.strip()
print(engagement_df.columns)

#rechecking for missing values (previously handled missing values in excel), detailed info in analysis report 
print(user_df.isnull().sum())
print(engagement_df.isnull().sum())

#convert non-datetime columns to datetime format 
engagement_df['last_session_creation_time'] = pd.to_datetime(engagement_df['last_session_creation_time'], errors='coerce')
print(engagement_df['last_session_creation_time'])

engagement_df['creation_time'] = pd.to_datetime(engagement_df['creation_time'], errors='coerce')
print(engagement_df['creation_time'])

#business metrics 

#defining adopted user, a user who has logged into the product on three separate days in at least one seven-­day period

#extrack week from dates
user_df['week'] = pd.to_datetime(user_df['time_stamp']).dt.to_period('W').apply(lambda r: r.start_time)

#number of logins per user per week 
weekly_logins = user_df.groupby(['user_id', 'week'])['time_stamp'].nunique()

#find and filter adopted users
adopted_users = weekly_logins[weekly_logins >= 3].reset_index()
adopted_users = adopted_users.groupby('user_id').size()
adopted_users_filtered = adopted_users[adopted_users > 0].index

adopted_users_df = user_df[user_df['user_id'].isin(adopted_users_filtered)]

#pd.set_option('display.max_rows', None)  # Remove limit on number of rows displayed
print(adopted_users_df.head())
#print(adopted_users_df)

#merge identified users with engagement_df
merged_df = pd.merge(engagement_df, user_df, on='user_id', how='left')

#identify adopted users with 1 and 0 for non-adopted users
merged_df['adopted'] = merged_df['user_id'].notna().astype(int)
merged_df['adopted'] = merged_df['user_id'].isin(adopted_users_filtered).astype(int)

#verify merged data 
merged_df.to_csv('/Users/georgetteturkson/Downloads/merged_data.csv', index=False)

#feature Engineering

#convert categorical variables to numerical values
categorical_features = ['creation_source']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(merged_df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

#convert datetime columns to numeric format
merged_df['last_session_creation_time'] = merged_df['last_session_creation_time'].astype(int) / 10**9  # Convert to seconds
merged_df['creation_time'] = merged_df['creation_time'].astype(int) / 10**9  # Convert to second

features = ['last_session_creation_time', 'enabled_for_marketing_drip', 'org_id', 'invited_by_user_id']
categorical_features = ['creation_source']

# Combine numerical features with one-hot encoded categorical features
X = pd.concat([merged_df[features], encoded_df], axis=1)
y = merged_df['adopted']


#train-test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#feature importance ranking
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

#download to mac
importance_df.to_csv('/Users/georgetteturkson/Downloads/feature_importance.csv', index=False)

#test accuracy of model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

#classification report 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

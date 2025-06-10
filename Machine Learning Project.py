import pandas as pd
import pickle
import streamlit as st
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

data_movies = pd.read_csv(r"C:\Users\Laptop inn\Desktop\Machine Larning project Material\tmdb_5000_movies.csv")
print(data_movies.head())

data_credits = pd.read_csv(r"C:\Users\Laptop inn\Desktop\Machine Larning project Material\tmdb_5000_credits.csv")
print(data_credits.head())

merged_data = data_movies.merge(data_credits, left_on='id', right_on='movie_id')
print(merged_data.head())

encoder = LabelEncoder()
merged_data["homepage"]=encoder.fit_transform(merged_data["homepage"])
merged_data["original_language"]=encoder.fit_transform(merged_data["original_language"])
merged_data["original_title"]=encoder.fit_transform(merged_data["original_title"])
merged_data["overview"]=encoder.fit_transform(merged_data["overview"])
merged_data["status"]=encoder.fit_transform(merged_data["status"])
merged_data["tagline"]=encoder.fit_transform(merged_data["tagline"])
merged_data["title_y"]=encoder.fit_transform(merged_data["title_y"])
merged_data["genres"] = merged_data["genres"].fillna('[]').astype(str)
merged_data["genres"]=encoder.fit_transform(merged_data["genres"])
merged_data["keywords"] = merged_data["keywords"].fillna('[]').astype(str)
merged_data["keywords"]=encoder.fit_transform(merged_data["keywords"])
merged_data["production_companies"] = merged_data["production_companies"].fillna('[]').astype(str)
merged_data["production_companies"]=encoder.fit_transform(merged_data["production_companies"])
merged_data["production_countries"] = merged_data["production_countries"].fillna('[]').astype(str)
merged_data["production_countries"]=encoder.fit_transform(merged_data["production_countries"])
merged_data["spoken_languages"] = merged_data["spoken_languages"].fillna('[]').astype(str)
merged_data["spoken_languages"]=encoder.fit_transform(data_movies["spoken_languages"])
merged_data["title_x"]=encoder.fit_transform(merged_data["title_x"])
merged_data["cast"] = merged_data["cast"].fillna('[]').astype(str)
merged_data["cast"]=encoder.fit_transform(merged_data["cast"])
merged_data["crew"] = merged_data["crew"].fillna('[]').astype(str)
merged_data["crew"]=encoder.fit_transform(merged_data["crew"])
print(merged_data.head())

x = data_movies.drop(columns='vote_count', axis = 1)
y = data_movies['vote_count']

x = x.fillna(0)

for col in x.columns:
   if x[col].dtype == 'object':
       x[col] = LabelEncoder().fit_transform(x[col].astype(str))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

clf = DecisionTreeRegressor()
clf.fit(x_train, y_train)

# ✅ Save the trained model
pickle.dump(clf, open('model.pkl', 'wb'))

# Now you can safely load it
model = pickle.load(open('model.pkl', 'rb'))


y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)
# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎬 Movie Vote Count Predictor")

# Movie features input
homepage = st.number_input("Homepage Encoded", 0)
original_language = st.number_input("Original Language Encoded", 0)
original_title = st.number_input("Original Title Encoded", 0)
overview = st.number_input("Overview Encoded", 0)
status = st.number_input("Status Encoded", 0)
tagline = st.number_input("Tagline Encoded", 0)
title = st.number_input("Title Encoded", 0)
genres = st.number_input("Genres Encoded", 0)
keywords = st.number_input("Keywords Encoded", 0)
production_companies = st.number_input("Production Companies Encoded", 0)
production_countries = st.number_input("Production Countries Encoded", 0)
spoken_languages = st.number_input("Spoken Languages Encoded", 0)
cast = st.number_input("Cast Encoded", 0)
crew = st.number_input("Crew Encoded", 0)

# Create feature vector
input_features = np.array([[homepage, original_language, original_title, overview, status,
                            tagline, title, genres, keywords, production_companies,
                            production_countries, spoken_languages, cast, crew]])

if st.button("Predict Vote Count"):
    prediction = model.predict(input_features)
    st.success(f"🎯 Predicted Vote Count: {int(prediction[0])}")
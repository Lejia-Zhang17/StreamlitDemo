import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.beta_container()
dataset = st.beta_container()
feature = st.beta_container()
model_training = st.beta_container()


st.markdown(
    """
    <style>
    .main{
    background-color:#F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
@st.cache
def get_data(filename):
    test_data = pd.read_csv(filename)

    return test_data
with header:
    st.title("Welcome to my awesome data science project!")
    # st.text('In this project I look into transactions of taxis in NYC')

with dataset:
    st.header('NYC text dataset')
    st.text("I found this dataset on blablablab.com")
    test_data=get_data("test_data.csv")

    st.write(test_data.head())

    st.subheader("Pick-up location ID distribution on the NYC dataset")
    station_dist = pd.DataFrame(test_data['start station id'].value_counts().head(50))
    st.bar_chart(station_dist)

with feature:
    st.header('The featrues I created')

    st.markdown('* **first feature: ** I created this feature becasue of this .. I calculated it using this logic ..')
    st.markdown('* **second feature: ** I created this feature becasue of this .. I calculated it using this logic ..')
with model_training:
    st.header("Time to train the model")
    st.text("Here you can choose the hyperparameters of the model and see how the performance changes!")

    sel_col, disp_col = st.beta_columns(2)

    max_depth=sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100,value=20,step=10)

    n_estimators = sel_col.selectbox("How many trees should there be?",options=[100,200,300,"No limit"],index=0)

    sel_col.text("Here is a list of features in my data")
    sel_col.write(test_data.columns)
    input_feature = sel_col.text_input("Which feature should be used as the input feature?","start station id")

    if n_estimators == "No limit":
        regr=RandomForestRegressor(max_depth)
    else:
        regr=RandomForestRegressor(max_depth,n_estimators=n_estimators)
    X=test_data[[input_feature]]
    y=test_data[['tripduration']]

    regr.fit(X,y)
    prediction = regr.predict(y)


    disp_col.subheader("Mean absolute error of the model is: ")
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader("Mean squared error of the model is: ")
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader("R squared error of the model is: ")
    disp_col.write(r2_score(y, prediction))


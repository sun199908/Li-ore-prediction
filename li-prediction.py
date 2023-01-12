from distutils.text_file import TextFile
from tkinter import _TakeFocusValue
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt 
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(page_title='Li ore prediction')

st. title("Li ore prediction app")
st.subheader('This app uses the ***Major element data of bauxite ***')
st.write('---')

df = pd.read_excel('锂矿大数据二分类2正式-3.xlsx')
X = df.drop('Li', axis=1)
y = df['Li']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)


st.sidebar.header('User Input Features')
analysis = st.sidebar.selectbox('Select a method',['Decision Tree','Random Forest'])


Al = st.sidebar.number_input('Al')
Si = st.sidebar.number_input('Si')
Al0Si = st.sidebar.number_input('Al0Si')
Fe = st.sidebar.number_input('Fe')

def user_input_features():
        data = {'Al':Al,
                'Si':Si,
                'Al0Si':Al0Si,
                'Fe':Fe,
                }
        features = pd.DataFrame(data, index=[0])
        return features

user_data = user_input_features()
st.subheader('**User Input parameters**')
st.write(user_data)

if analysis == 'Decision Tree':
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    user_result = DT.predict(user_data)
    st.title('')
    st.subheader('**Conclusion:**')
    pred_button = st.button('Predict')
    if pred_button:
        if user_result == 1:
            st.write('低于锂矿边界品位')
        elif user_result == 3:
            st.write('达到锂矿边界品位')
else:
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    user_result = RF.predict(user_data)
    st.title('')
    st.subheader('**Conclusion:**')
    pred_button = st.button('Predict')
    if pred_button:
        if user_result == 1:
            st.write('低于锂矿边界品位')
        elif user_result == 3:
            st.write('达到锂矿边界品位')
        



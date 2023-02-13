import streamlit as st
import numpy as np
import pandas as pd
import altair as alt 
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
analysis = st.sidebar.selectbox('Select a method',['KNeighbors','SVM'])


Al = st.sidebar.number_input('Al2O3')
Si = st.sidebar.number_input('SiO2')
Al0Si = st.sidebar.number_input('Al2O3/SiO2')
Fe = st.sidebar.number_input('TFeO')

def user_input_features():
        data = {'Al2O3':Al,
                'SiO2':Si,
                'Al2O3/SiO2':Al0Si,
                'TFeO':Fe,
                }
        features = pd.DataFrame(data, index=[0])
        return features

user_data = user_input_features()
st.subheader('**User Input parameters**')
st.write(user_data)

if analysis == 'KNeighbors':
    DT = KNeighborsClassifier()
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
    RF = SVM()
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
        



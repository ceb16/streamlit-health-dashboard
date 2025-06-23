# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import inspect
from streamlit_lottie import st_lottie
from numerize import numerize
from itertools import chain
import plotly.graph_objects as go
import plotly.express as px
import joblib
import statsmodels.api as sm
import sklearn
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(layout="wide", page_title="Cardiovascular Disease Introductory Analysis")

# Automatically load dataset from local path
try:
    data_ha = pd.read_csv("cardio_train_c.csv")
except FileNotFoundError:
    st.error("The dataset 'cardio_train_c.csv' was not found in the project folder. Please make sure it exists.")
    st.stop()

# Sidebar Navigation
menu_id = st.sidebar.radio("Navigation Bar", ["Home", "EDA", "Analyses", "Conclusion"])

# Home Page
if menu_id == "Home":
    st.title("Cardiovascular Disease Dashboard")
    st.write("""
        This dashboard provides visual insights into a dataset on cardiovascular health extracted from kaggle,
        including demographic breakdowns, health indicators, and patterns related to disease presence. You can find a preview of the dataset as follows:
    """)
    st.dataframe(data_ha.head())

# EDA Page
elif menu_id == "EDA":
    col1, col2, col3 = st.columns([3, 3, 3])
    number_of_males = len(data_ha[data_ha["gender"] == 2])
    number_of_females = len(data_ha[data_ha["gender"] == 1])
    data_length = len(data_ha)

    with col1:
        st.metric("Observations", f"{data_length:,}")
    with col2:
        st.metric("Number of Male", f"{number_of_males:,}")
    with col3:
        st.metric("Number of Female", f"{number_of_females:,}")

    # Age bracket bar chart
    age_counts = data_ha.groupby('age_bracket_number')['age_in_years'].count().reset_index()
    fig = px.bar(age_counts, x='age_bracket_number', y='age_in_years')
    fig.update_layout(xaxis_title='Age Bracket', yaxis_title='Count of Ages', title='Age Distribution')
    st.plotly_chart(fig, use_container_width=True)

    # Pie chart for cardio
    st.markdown("### Cardiovascular Disease Proportion")
    cardio_counts = data_ha['cardio'].value_counts().reset_index()
    cardio_counts.columns = ['Cardio', 'Count']
    fig = px.pie(cardio_counts, values='Count', names='Cardio')
    st.plotly_chart(fig)

    # Blood pressure box plot
    blood_pressure = data_ha[['ap_hi', 'ap_lo']]
    fig = px.box(blood_pressure, title='Systolic vs Diastolic Blood Pressure')
    st.plotly_chart(fig)

    # Glucose and cardio stacked bar
    grouped = data_ha.groupby(['gluc', 'cardio']).size().unstack()
    fig = go.Figure()
    gluc_labels = ['Normal', 'Above Normal', 'Well Above Normal']
    for i, category in enumerate(grouped.index):
        fig.add_trace(go.Bar(
            x=['No', 'Yes'],
            y=grouped.loc[category],
            name=f'gluc {gluc_labels[i]}'
        ))
    fig.update_layout(title='Cardiovascular Disease by Glucose Category', barmode='stack')
    st.plotly_chart(fig)

# Analyses Page
elif menu_id == "Analyses": 
    st.markdown("### Analysis by Lifestyle Factors")
    features = ['smoke', 'alco', 'active']
    col1, col2, col3 = st.columns(3)
    for idx, feature in enumerate(features):
        counts = data_ha.groupby([feature, 'cardio']).size().unstack(fill_value=0)
        fig = go.Figure()
        for col in counts.columns:
            fig.add_trace(go.Bar(x=['No', 'Yes'], y=counts[col], name=str(col).replace('0', 'No').replace('1', 'Yes')))
        fig.update_layout(
            barmode='stack',
            xaxis_title=feature,
            yaxis_title='Count',
            title=f'{feature.capitalize()} vs. Cardiovascular Disease'
        )
        if idx % 3 == 0:
            col1.plotly_chart(fig)
        elif idx % 3 == 1:
            col2.plotly_chart(fig)
        else:
            col3.plotly_chart(fig)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.scatter(data_ha, x='bmi', y='gender', color='cardio', title='BMI and Gender Patterns')
        st.plotly_chart(fig)

    with col2:
        fig = px.scatter(data_ha, x='age_in_years', y='bmi', color='cardio',
                         title='BMI vs Age by Cardio Health', color_discrete_sequence=['red', 'green'])
        st.plotly_chart(fig)

    with col1:
        group_means = data_ha.groupby(['gender', 'cardio']).agg({'pulse': 'mean'}).reset_index()
        group_means.rename(columns={'pulse': 'Mean Pulse'}, inplace=True)
        group_means['Gender'] = group_means['gender'].map({1: 'Male', 2: 'Female'})
        fig = go.Figure(data=[
            go.Bar(name='Male', x=group_means[group_means['Gender'] == 'Male']['cardio'], y=group_means[group_means['Gender'] == 'Male']['Mean Pulse']),
            go.Bar(name='Female', x=group_means[group_means['Gender'] == 'Female']['cardio'], y=group_means[group_means['Gender'] == 'Female']['Mean Pulse'])
        ])
        fig.update_layout(
            xaxis_title='Cardio',
            yaxis_title='Mean Pulse',
            title='Mean Pulse by Gender and Cardio Status',
            barmode='group'
        )
        st.plotly_chart(fig)

# Conclusion Page
elif menu_id == "Conclusion":
    st.markdown("### Conclusion")
    st.write("""
         This dashboard highlights a multifactorial pattern behind cardiovascular disease. Elevated blood pressure, high glucose levels, smoking, drinking, and high BMI—all show strong associations with heart conditions. Age and gender also play a role, but lifestyle and metabolic indicators appear more predictive. While activity is generally beneficial, it alone doesn't offset risk if other factors are present. Understanding these relationships is crucial for designing effective health interventions and predictive models. Use these insights to inform public health policies
        and individual risk assessments.
    """)

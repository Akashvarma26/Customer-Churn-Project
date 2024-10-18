import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow
model = tensorflow.keras.models.load_model("ann1.h5")
ohe_geo=pickle.load(open("ohe_geo.pkl","rb"))
le_gender=pickle.load(open("le_gender.pkl","rb"))
sc=pickle.load(open("sc.pkl","rb"))

st.title("Customer Churn Predictor")

st.header("Hello")

st.write("This app is used to predict whether a customer is likely to churn. We will use a simple Artificial Neural Network (ANN) algorithm to predict the probability of churn.\n \n")

st.text("Please fill the fields below to predict.")

geography=st.selectbox('Geography',ohe_geo.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider("Tenure",1,4)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card? ",[0,1])
is_active_member=st.selectbox("Is Active Member? ",[0,1])

input_data= pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_enco=ohe_geo.transform([[geography]])
geo_enc_df=pd.DataFrame(geo_enco.toarray(),columns=ohe_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_enc_df],axis=1)

input_scaled=sc.transform(input_data)
pred_prob=model.predict(input_scaled)[0][0]

if pred_prob>0.5:
    st.write(f'Customer is likely to Churn with probability of {pred_prob:.3f}')
else:
    st.write(f'Customer is not likely to Churn with probability of {pred_prob:.3f}')

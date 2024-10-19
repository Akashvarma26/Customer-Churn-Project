import streamlit as st
import pandas as pd
import pickle
import tensorflow

model = tensorflow.keras.models.load_model("models/ann1.h5")
ohe_geo = pickle.load(open("preprocessing/ohe_geo.pkl", "rb"))
le_gender = pickle.load(open("preprocessing/le_gender.pkl", "rb"))
sc = pickle.load(open("preprocessing/sc.pkl", "rb"))

st.title("Customer Churn Predictor developed by Akashvarma26")

name = st.text_input("Please enter your name:")

if name:
    st.header(f"Hello, {name}! 👋")
    st.write(f"Welcome to the Customer Churn Prediction app, {name}.This app is used to predict whether a customer is likely to churn. A simple Artificial Neural Network (ANN) algorithm is used to predict the probability of churn.\n \n")
    st.text("Please fill out the following details of the customer")
    geography = st.selectbox('Geography', ohe_geo.categories_[0])
    gender = st.radio('Gender', le_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance', min_value=0.0, format="%.2f")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
    tenure = st.slider("Tenure", 0, 10)
    num_of_products = st.slider("Number of Products", 1, 4)
    has_cr_card = st.radio("Has Credit Card?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.radio("Is Active Member?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_enco = ohe_geo.transform([[geography]])
    geo_enc_df = pd.DataFrame(geo_enco.toarray(), columns=ohe_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_enc_df], axis=1)
    input_scaled = sc.transform(input_data)
    pred_prob = model.predict(input_scaled)[0][0]
    pred_percent = int(pred_prob*100)
    st.subheader(f"Churn Prediction Result of the customer:")
    st.progress(pred_percent)
    if pred_prob>0.5:
        st.error(f'⚠️ {name}, the customer is likely to churn. The churn probability is {pred_prob:.3f}.')
    else:
        st.success(f'✅ {name}, the customer is not likely to churn. The churn probability is {pred_prob:.3f}.')
    if credit_score<500:
        st.warning(f"{name}, the credit score is quite low. Customers with low credit scores might have higher churn probabilities.")
    if balance==0:
        st.info(f"{name}, the balance is zero. Customers with no account balance might have different churn behavior.")
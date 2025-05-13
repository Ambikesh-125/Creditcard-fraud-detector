import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
data = pd.read_csv('creditcard copy.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit Web App
st.title("💳 Credit Card Fraud Detection")

st.markdown("### Model Accuracy")
st.write(f"✅ Training Accuracy: **{train_acc:.2f}**")
st.write(f"🧪 Testing Accuracy: **{test_acc:.2f}**")

st.markdown("### Predict a Transaction")
st.info("Enter **30 comma-separated numeric values** (e.g., `-1.23, 2.34, 0.01, ...`).")

input_df = st.text_input('Enter features:', '')

submit = st.button("Submit")

if submit:
    raw_values = input_df.split(',')
    # Clean values: strip whitespace and quotes
    cleaned_values = [x.strip().strip('"').strip("'") for x in raw_values if x.strip() != '']

    invalid_values = []
    for x in cleaned_values:
        try:
            float(x)
        except ValueError:
            invalid_values.append(x)

    if invalid_values:
        st.error(f"❌ Invalid entries detected: {invalid_values}")
        st.warning("Please enter only numeric values (no letters, symbols, or quotes).")
    elif len(cleaned_values) != X.shape[1]:
        st.error(f"❌ Expected {X.shape[1]} features, but got {len(cleaned_values)}.")
    else:
        try:
            features = np.array([float(x) for x in cleaned_values])
            prediction = model.predict(features.reshape(1, -1))
            if prediction[0] == 0:
                st.success("✅ Legitimate transaction")
            else:
                st.error("⚠️ Fraudulent transaction detected!")
        except Exception as e:
            st.error("An error occurred during prediction.")
            st.exception(e)

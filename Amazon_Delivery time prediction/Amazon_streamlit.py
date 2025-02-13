import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Delivery_amazon.csv")
    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    df.dropna(inplace=True)  # Remove missing values
    return df.sample(frac=0.5, random_state=42)  # Reduce dataset size

df = load_data()
st.set_page_config(page_title="Regression Model Dashboard", layout="wide")
st.title("📊 Regression Model Dashboard")
st.markdown("---")
st.write("### Amazon Delivery Prediction Models")

# Sidebar Configuration
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose a Regression Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])

# Data Preprocessing
X = df.drop(columns=["Delivery_Time"]) if "Delivery_Time" in df.columns else df
y = df["Delivery_Time"] if "Delivery_Time" in df.columns else None

# Apply scaling only for Linear Regression
if model_choice == "Linear Regression":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=50, random_state=42)  # Optimized estimators
elif model_choice == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Display Metrics in Sidebar
st.sidebar.markdown("### Model Performance Metrics")
st.sidebar.write(f"**RMSE:** {rmse:.2f}")
st.sidebar.write(f"**R² Score:** {r2:.2f}")
st.sidebar.write(f"**MAE:** {mae:.2f}")

# Interactive Data Summary
st.write("### Data Overview")
st.dataframe(df.head())

# Feature Importance (For Tree-Based Models)
if model_choice in ["Random Forest", "Gradient Boosting"]:
    feature_importances = model.feature_importances_
    feature_names = df.drop(columns=["Delivery_Time"]).columns if "Delivery_Time" in df.columns else df.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    
    st.write("### Feature Importance")
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], ax=ax_imp)
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)

# Plot Predictions
st.write("### Predictions vs Actual")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, label="Predicted vs Actual")
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Fit")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")
ax.legend()
st.pyplot(fig)

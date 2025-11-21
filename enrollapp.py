import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from prophet import Prophet

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="Rural School Enrollment Dashboard", layout="wide")

# Custom CSS for larger fonts and buttons
st.markdown("""
    <style>
    body, .stText, .stMarkdown, .stNumberInput, .stSelectbox {
        font-size: 18px !important;
    }
    div.stButton > button:first-child {
        font-size:22px !important;
        font-weight:bold !important;
        height:3em !important;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📊 Rural School Enrollment Prediction and Dashboard")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your school data [CSV format with Year, District, Enrollment]", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file).dropna()
    required_columns = ["Year", "District", "Enrollment"]
    
    if not all(col in data.columns for col in required_columns):
        st.error(f"CSV must contain columns: {required_columns}")
    else:
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Dashboard", "🔍 Model Insights"])

        # ---------------- TAB 1: PREDICTION ----------------
        with tab1:
            st.subheader("Enter details to predict enrollment")

            year = st.number_input("Enter Year", min_value=int(data["Year"].min()), value=int(data["Year"].max()) + 1)
            district = st.selectbox("Select District", data["District"].unique())
            current_teachers = st.number_input("Enter current number of Teachers", min_value=0, value=0)
            current_classrooms = st.number_input("Enter current number of Classrooms", min_value=0, value=0)
            students_per_teacher = st.number_input("Students per Teacher (default 30)", min_value=1, value=30)
            students_per_classroom = st.number_input("Students per Classroom (default 40)", min_value=1, value=40)

            if st.button("🚀 Predict Enrollment", use_container_width=True):
                # Filter district data and prepare for Prophet
                district_data = data[data["District"] == district][["Year", "Enrollment"]].rename(columns={"Year": "ds", "Enrollment": "y"})
                
                # Prophet model with linear growth and adjustable trend sensitivity
                model = Prophet(yearly_seasonality=True, growth='linear', changepoint_prior_scale=0.5)
                model.fit(district_data)
                
                future = pd.DataFrame({"ds": [pd.Timestamp(year, 1, 1)]})
                forecast = model.predict(future)
                prediction = max(0, int(forecast["yhat"].values[0]))  # Ensure no negative enrollment

                # Teacher & classroom calculations
                teachers_needed = int(np.ceil(prediction / students_per_teacher))
                classrooms_needed = int(np.ceil(prediction / students_per_classroom))
                teachers_to_add = max(0, teachers_needed - current_teachers)
                classrooms_to_add = max(0, classrooms_needed - current_classrooms)

                # Add more / enough messages
                teachers_text = f'<span style="color:red; font-size:22px; font-weight:bold;">Add {teachers_to_add} more</span>' if teachers_to_add > 0 else '<span style="color:green; font-size:22px; font-weight:bold;">✅ Enough teachers</span>'
                classrooms_text = f'<span style="color:red; font-size:22px; font-weight:bold;">Add {classrooms_to_add} more</span>' if classrooms_to_add > 0 else '<span style="color:green; font-size:22px; font-weight:bold;">✅ Enough classrooms</span>'

                # Display results with bigger fonts and centered layout
                st.markdown(
                    f"""
                    <div style="border:2px solid #ccc; border-radius:10px; padding:25px; background-color:#d4edda; text-align:center;">
                        <h2 style="color:#155724; font-size:28px;">📌 Predicted Enrollment for {year} ({district})</h2>
                        <h1 style="color:#d9534f; font-size:60px; margin:20px 0;">{prediction}</h1>
                        <hr style="border-top:2px solid #155724; width:50%;">
                        <p style="font-size:22px; margin:10px;">👩‍🏫 Teachers Required: {teachers_text}</p>
                        <p style="font-size:22px; margin:10px;">🏫 Classrooms Required: {classrooms_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------------- TAB 2: DASHBOARD ----------------
        with tab2:
            st.subheader("Enrollment Over the Years")
            fig, ax = plt.subplots()
            for dist in data["District"].unique():
                dist_data = data[data["District"] == dist]
                ax.plot(dist_data["Year"], dist_data["Enrollment"], marker="o", label=dist)
            ax.set_xlabel("Year", fontsize=16)
            ax.set_ylabel("Enrollment", fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=14)
            fig.tight_layout()
            st.pyplot(fig)

        # ---------------- TAB 3: MODEL INSIGHTS ----------------
        with tab3:
            st.subheader("Prophet Trend Forecast Example")
            selected_district = st.selectbox("Select district for forecast plot", data["District"].unique())
            dist_data = data[data["District"] == selected_district][["Year", "Enrollment"]].rename(columns={"Year": "ds", "Enrollment": "y"})

            # Prophet trend
            model = Prophet(yearly_seasonality=True, growth='linear', changepoint_prior_scale=0.5)
            model.fit(dist_data)
            future = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future)

            fig2 = model.plot(forecast)
            st.pyplot(fig2)

            st.markdown("---")
            st.subheader("📊 Feature Importance")

            # Extra features for RandomForest
            feature_cols = [col for col in data.columns if col not in ["Enrollment", "Year", "District"]]

            if len(feature_cols) > 0:
                X = data[feature_cols]
                y = data["Enrollment"]
                X_encoded = pd.get_dummies(X, drop_first=True)

                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_encoded, y)

                importance = pd.DataFrame({
                    "Feature": X_encoded.columns,
                    "Importance": rf_model.feature_importances_
                }).sort_values(by="Importance", ascending=True)

                fig3, ax = plt.subplots(figsize=(10, max(3, len(importance)*0.4)))
                ax.barh(importance["Feature"], importance["Importance"], color="skyblue")
                ax.set_xlabel("Importance", fontsize=16)
                ax.set_title("Feature Importance", fontsize=18)
                ax.tick_params(axis='y', labelsize=14)
                fig3.tight_layout()
                st.pyplot(fig3)
            else:
                st.info("No extra features found for Feature Importance analysis.")

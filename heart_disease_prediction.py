import streamlit as st
import pandas as pd
import joblib

# Load the trained model
loaded_rf_model = joblib.load('best_rf_model.joblib')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="💓",
    layout="wide",
)

# Add a title to the left of the image
st.markdown("<h1 style='text-align: left; color: #0000ff; font-weight: bold;'>Heart Disease Predictor</h1>", unsafe_allow_html=True)

# Add logo
st.image('heart_logo.jpg', width=200)  # Adjust width as needed

# Define function to get user input
def get_user_input():
    with st.sidebar:
        st.title('User Input Features')

        # Set background color for the sidebar
        st.markdown(
            """
            <style>
            ::-webkit-scrollbar {
                width: 10px;
            }
            ::-webkit-scrollbar-thumb {
                background-color: #e6e6fa;
                border-radius: 5px;
            }
            .sidebar-content {
                background-color: #e6e6fa;  /* Light Purple */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        form = st.form(key='user_input_form')
        slope_of_peak_exercise_st_segment = form.slider('Slope of Peak Exercise ST Segment', 1, 3, 2)
        thal = form.selectbox('Thallium Stress Test Result', ['normal', 'fixed_defect', 'reversible_defect'])
        resting_blood_pressure = form.slider('Resting Blood Pressure', 90, 200, 120)
        chest_pain_type = form.slider('Chest Pain Type', 1, 4, 2)
        num_major_vessels = form.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 3, 1)
        fasting_blood_sugar_gt_120_mg_per_dl = form.checkbox('Fasting Blood Sugar > 120 mg/dl')
        resting_ekg_results = form.selectbox('Resting EKG Results', [0, 1, 2])
        serum_cholesterol_mg_per_dl = form.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
        oldpeak_eq_st_depression = form.slider('Oldpeak Eq ST Depression', 0.0, 6.0, 2.0)
        sex = form.radio('Sex', ['female', 'male'])
        age = form.slider('Age', 29, 77, 50)
        max_heart_rate_achieved = form.slider('Max Heart Rate Achieved', 80, 200, 150)
        exercise_induced_angina = form.checkbox('Exercise-Induced Angina')

        submitted = form.form_submit_button('Predict', on_click=submit_button_click)
       
    if submitted:
        user_input = {
            'slope_of_peak_exercise_st_segment': slope_of_peak_exercise_st_segment,
            'thal': thal,
            'resting_blood_pressure': resting_blood_pressure,
            'chest_pain_type': chest_pain_type,
            'num_major_vessels': num_major_vessels,
            'fasting_blood_sugar_gt_120_mg_per_dl': fasting_blood_sugar_gt_120_mg_per_dl,
            'resting_ekg_results': resting_ekg_results,
            'serum_cholesterol_mg_per_dl': serum_cholesterol_mg_per_dl,
            'oldpeak_eq_st_depression': oldpeak_eq_st_depression,
            'sex': 0 if sex == 'female' else 1,
            'age': age,
            'max_heart_rate_achieved': max_heart_rate_achieved,
            'exercise_induced_angina': exercise_induced_angina
        }

        return pd.DataFrame([user_input])

# Function to handle submit button click
def submit_button_click():
    # Add any additional actions on submit
    pass

# Get user input
user_df = get_user_input()

# If user has submitted input, perform prediction
if user_df is not None:
    # Ensure that the user input DataFrame has the same features as the training data
    user_df = user_df.reindex(columns=['slope_of_peak_exercise_st_segment', 'thal_fixed_defect', 'thal_normal', 'thal_reversible_defect',
                                       'resting_blood_pressure', 'chest_pain_type', 'num_major_vessels',
                                       'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results',
                                       'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex', 'age',
                                       'max_heart_rate_achieved', 'exercise_induced_angina'], fill_value=0)

    # Make predictions
    prediction = loaded_rf_model.predict(user_df)
    prediction_proba = loaded_rf_model.predict_proba(user_df)[:, 1]

    # Display prediction results in a table
    st.subheader('Prediction Results')
    results_df = pd.DataFrame({
        'Prediction': ["Positive" if prediction[0] == 1 else "Negative"],
        'Probability': [f'{prediction_proba[0]:.4f}']
    })
    st.table(results_df)

    # Display a note about model limitations
    model_limitations_note = """
    ## Important Note
    The model has been trained on a limited dataset of 180 records. As a result, its accuracy may vary, and it can make mistakes.
    Please consider this information when interpreting the predictions.
    """
    st.markdown(model_limitations_note, unsafe_allow_html=True)


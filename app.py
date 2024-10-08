import streamlit as st
import numpy as np
import tensorflow as tf


# Define the function to load the model
@st.cache_resource
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


# Constants
MASS_ATTENUATION_COEFF = 0.02  # cm^2/g
R_INITIAL = 12.0
DEFAULT_AIR_DENSITY = 0.0294  # g/cm^2
ACQUISITION_TIME = 5.0
DEFAULT_V = 1.0  # Default speed of the detector
DEFAULT_BRANCHING_RATIO = 0.8519  # Constant default branching ratio


# Define a function to adjust the background count rate
def adjust_background_count_rate(background_count_rate, alarm_threshold_factor=1.5):
    min_rate = 0.0
    max_rate = 8000.0
    scaled_background_count_rate = min(max(min_rate, background_count_rate), max_rate)
    adjusted_background_count_rate = scaled_background_count_rate * alarm_threshold_factor
    return adjusted_background_count_rate


# Define the calculate_immobile_MDD function
def calculate_immobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, air_density, v, acquisition_time, background_count_rate = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate)

    R_min = 0
    R_max = 10000
    ALARM_LEVEL = 1.0
    DEFAULT_ACTIVITY = 26000

    while (R_max - R_min) > tolerance:
        R_fix = (R_min + R_max) / 2
        current_params = (
            Pdesired, DEFAULT_ACTIVITY, DEFAULT_BRANCHING_RATIO, air_density, v, acquisition_time, adjusted_background_count_rate,
            ALARM_LEVEL, R_fix)

        angles_rad = np.deg2rad(angles)
        angles_rad = tf.convert_to_tensor(angles_rad[:, np.newaxis], dtype=tf.float32)

        # Get model predictions
        model_output = model(angles_rad)

        # Ensure model output is a tensor
        if isinstance(model_output, dict) and 'output' in model_output:
            rel_eff = model_output['output']
        elif isinstance(model_output, tf.Tensor):
            rel_eff = model_output
        else:
            raise ValueError("Unexpected format for model output.")

        # Ensure the shape is as expected
        if rel_eff.shape != (angles_rad.shape[0], 1):
            raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

        fluence_rate = DEFAULT_ACTIVITY * DEFAULT_BRANCHING_RATIO * rel_eff
        detection_probability = 1 - tf.math.exp(-fluence_rate * R_fix * air_density / adjusted_background_count_rate)
        mean_detection_probability = tf.reduce_mean(detection_probability).numpy()

        print(
            f"R_min: {R_min}, R_max: {R_max}, R_fix: {R_fix}, Mean Detection Probability: {mean_detection_probability}")

        if mean_detection_probability >= Pdesired:
            R_max = R_fix
        else:
            R_min = R_fix

    return R_fix


# Define the calculate_mobile_MDD function
def calculate_mobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, air_density, v, acquisition_time, background_count_rate = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate)

    R_min = 0
    R_fix = calculate_immobile_MDD(model, angles, params, tolerance=tolerance)  # Starting point for mobile MDD
    R_max = 2 * R_fix  # Initial doubling of R_fix

    DEFAULT_ACTIVITY = 26000

    while (R_max - R_min) > tolerance:
        R_test = (R_max + R_min) / 2
        total_detection_prob = 0.0
        distance_traveled = 0.0
        current_distance_per_step = v * ACQUISITION_TIME

        step_count = 0
        while distance_traveled < R_test:
            angle_rad = np.arcsin(distance_traveled / R_test)
            angles_rad = tf.convert_to_tensor([[angle_rad]], dtype=tf.float32)

            # Get model predictions
            model_output = model(angles_rad)

            # Ensure model output is a tensor
            if isinstance(model_output, dict) and 'output' in model_output:
                rel_eff = model_output['output']
            elif isinstance(model_output, tf.Tensor):
                rel_eff = model_output
            else:
                raise ValueError("Unexpected format for model output.")

            # Ensure the shape is as expected
            if rel_eff.shape != (angles_rad.shape[0], 1):
                raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

            fluence_rate = DEFAULT_ACTIVITY * DEFAULT_BRANCHING_RATIO * rel_eff
            detection_prob = 1 - tf.math.exp(-fluence_rate * R_test * air_density / adjusted_background_count_rate)
            total_detection_prob += detection_prob
            distance_traveled += current_distance_per_step

            step_count += 1
            if step_count > 10000:  # Break if too many steps to prevent infinite loop
                print(f"Breaking loop at step {step_count} to prevent infinite loop.")
                break

        mean_total_detection_prob = tf.reduce_mean(total_detection_prob).numpy()
        print(f"R_test: {R_test}, Mean Total Detection Probability: {mean_total_detection_prob}, Speed: {v}, Steps: {step_count}")

        if mean_total_detection_prob >= Pdesired:
            R_max = R_test
        else:
            R_min = R_test

    return R_test


# Define a function to calculate the maximum detectable distance (MDD)
def calculate_mdd(model, angles, params, is_mobile, tolerance):
    if is_mobile:
        return calculate_mobile_MDD(model, angles, params, tolerance)
    else:
        return calculate_immobile_MDD(model, angles, params, tolerance)

# Streamlit UI
st.title("Maximum Detectable Distance (MDD) Calculator")

Pdesired = st.slider("Desired Detection Probability", 0.0, 1.0, 0.95)
angles_input = st.text_input("Angles (comma separated)", "0, 10, 20, 30")

# Input fields for the parameters
col1, col2 = st.columns(2)

with col1:
    background_count_rate = st.number_input("Background Count Rate (CPM)", value=600)
    is_mobile = st.checkbox("Is the detector mobile?")

    if is_mobile:
        v = st.number_input("Speed of the Detector/Vehicle (m/s)", value=DEFAULT_V)
    else:
        v = DEFAULT_V

tolerance = 1e-3

# Prepare the parameters
angles = [float(angle) for angle in angles_input.split(",")]
params = (Pdesired, 26000, 0.0294, v, 1.0, background_count_rate)

# Load the model
model_path = 'my_model.tf'  #  path to model directory
model = load_model(model_path)

# Calculate MDD
if st.button("Calculate MDD"):
    try:
        MDD = calculate_mdd(model, angles, params, is_mobile, tolerance)
        st.write(f"Maximum Detectable Distance (MDD): {MDD:.2f} inches")
    except ValueError as ve:
        st.error(str(ve))

import pandas as pd
import numpy as np

def load_and_clean_sensor(file_path, sensor_empty, smoothing_window=2, speed_threshold=1.0):
    """
    Data cleaning and filtering pipeline for the Suspension Sensor PINN.
    Smooths raw signals, converts raw sensor reads to deflection, and removes physical anomalies.
    """
    df = pd.read_csv(file_path)
    
    # Smooth raw signals to prevent noisy derivatives in autograd
    df['Speed_Smooth'] = df['Speed (m/s)'].rolling(window=smoothing_window, min_periods=1).mean()
    df['Sensor_Smooth'] = df['Sensor 1 read'].rolling(window=smoothing_window, min_periods=1).mean()
    
    # Speed filter
    df = df[df['Speed_Smooth'] > speed_threshold].copy()
    
    # Convert suspension sensor read (mm) to vertical deflection (m)
    df['Deflection_m'] = (sensor_empty - df['Sensor_Smooth']) / 1000.0
    
    # Remove out-of-bound physical outliers
    df = df[(df['Deflection_m'] > -0.05) & (df['Deflection_m'] < 0.15)]
    return df


def load_and_clean_obd_peugeot(file_path):
    """
    Data cleaning and traction masking pipeline for the Peugeot Partner OBD model.
    """
    df = pd.read_csv(file_path)
    
    # Handle missing signals and smooth inputs
    df['mf'] = df['mf'].ffill().bfill()
    df['Theta (rad)'] = df['Theta (rad)'].ffill().fillna(0.0)
    df['Fuel_L_s_real'] = df['Fuel flow (L/s)'].rolling(window=15, min_periods=1).mean()
    df['Speed_Smooth'] = df['Speed (m/s)'].rolling(window=5, min_periods=1).mean()
    
    # Compute vehicle acceleration
    df['Delta_V'] = df['Speed_Smooth'].diff()
    df['Delta_T'] = df['Acum time (s)'].diff().replace(0, np.nan)
    df['acceleration_approx'] = (df['Delta_V'] / df['Delta_T']).fillna(0.0)
    
    # Traction Mask: only enforce physical loss when vehicle is accelerating above a speed limit
    df['Mask'] = ((df['acceleration_approx'] > 0.1) & (df['Speed_Smooth'] > 2.0)).astype(float)
    
    df = df[df['Speed_Smooth'] > 1.0].copy()
    return df


def load_and_clean_obd_tracto(file_path):
    """
    Data cleaning pipeline for the Tracto Freightliner OBD model.
    """
    df = pd.read_excel(file_path)
    
    df['mf'] = pd.to_numeric(df['mf'], errors='coerce')
    df['Fuel flow (L/s)'] = pd.to_numeric(df['Fuel flow (L/s)'], errors='coerce')
    
    # Smooth signals
    df['Theta (rad)'] = df['Theta (rad)'].ffill().fillna(0.0)
    df['Fuel_L_s_real'] = df['Fuel flow (L/s)'].rolling(window=20, min_periods=1).mean()
    df['Speed_Smooth'] = df['Speed (m/s)'].rolling(window=20, min_periods=1).mean()
    
    return df


def evaluate_tripwire(window, vehicle="peugeot"):
    """
    Validates a data window using experimental 'Cable Trampa' (Tripwire Logic) filters.
    Returns True if the window is valid, False if it should be discarded.
    """
    acceleration_approx = window['Speed_Smooth'].diff().fillna(0)
    
    if vehicle == "peugeot" or vehicle == "ram":
        # 1. Fuel Flow limit (anomalies / signal spikes)
        fuel_cond = (window['Fuel flow (L/s)'] > 0.006).any()
        # 2. Inertia coefficient mf limit (clutching/shifting noise)
        mf_cond = (window['mf'] > 30.0).any()
        # 3. Low velocity limit (idling, combustion without traction work)
        speed_cond = window['Speed_Smooth'].mean() < 3.0
        # 4. Low cargo limit (empty or near-empty payloads are neglected)
        payload_cond = window['Carga (kg)'].mean() < 70.0
        # 5. Traction vs Brake filter (expects accelerating/cruising state at least 60% of time)
        traction_cond = (acceleration_approx >= -0.1).mean() < 0.60
        # 6. Topographical noise
        topography_cond = window['Theta (rad)'].std() > 0.1
        
        if fuel_cond or mf_cond or speed_cond or payload_cond or traction_cond or topography_cond:
            return False
            
    elif vehicle == "tracto":
        # Heavily calibrated tripwires for large diesel engine tractocamión
        fuel_cond = (window['Fuel flow (L/s)'] > 0.05).any()
        mf_cond = (window['mf'] > 80.0).any()
        speed_cond = window['Speed_Smooth'].mean() < 3.0
        payload_cond = window['Carga (kg)'].mean() < 1000.0  # limit set to 1 ton
        traction_cond = (acceleration_approx >= -0.1).mean() < 0.60
        topography_cond = window['Theta (rad)'].std() > 0.1
        
        if fuel_cond or mf_cond or speed_cond or payload_cond or traction_cond or topography_cond:
            return False
            
    return True

import pandas as pd
import numpy as np

def cargar_y_limpiar_sensor(file_path, sensor_vacio, smoothing_window=2, speed_threshold=1.0):
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
    df['Deflexion_m'] = (sensor_vacio - df['Sensor_Smooth']) / 1000.0
    
    # Remove out-of-bound physical outliers
    df = df[(df['Deflexion_m'] > -0.05) & (df['Deflexion_m'] < 0.15)]
    return df


def cargar_y_limpiar_obd_peugeot(file_path):
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
    df['Aceleracion_aprox'] = (df['Delta_V'] / df['Delta_T']).fillna(0.0)
    
    # Traction Mask: only enforce physical loss when vehicle is accelerating above a speed limit
    df['Mascara'] = ((df['Aceleracion_aprox'] > 0.1) & (df['Speed_Smooth'] > 2.0)).astype(float)
    
    df = df[df['Speed_Smooth'] > 1.0].copy()
    return df


def cargar_y_limpiar_obd_tracto(file_path):
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


def evaluar_cable_trampa(ventana, vehiculo="peugeot"):
    """
    Validates a data window using experimental 'Cable Trampa' (Tripwire Logic) filters.
    Returns True if the window is valid, False if it should be discarded.
    """
    aceleracion_aprox = ventana['Speed_Smooth'].diff().fillna(0)
    
    if vehiculo == "peugeot" or vehiculo == "ram":
        # 1. Fuel Flow limit (anomalies / signal spikes)
        cond_combustible = (ventana['Fuel flow (L/s)'] > 0.006).any()
        # 2. Inertia coefficient mf limit (clutching/shifting noise)
        cond_mf = (ventana['mf'] > 30.0).any()
        # 3. Low velocity limit (idling, combustion without traction work)
        cond_velocidad = ventana['Speed_Smooth'].mean() < 3.0
        # 4. Low cargo limit (empty or near-empty payloads are neglected)
        cond_carga = ventana['Carga (kg)'].mean() < 70.0
        # 5. Traction vs Brake filter (expects accelerating/cruising state at least 60% of time)
        cond_traccion = (aceleracion_aprox >= -0.1).mean() < 0.60
        # 6. Topographical noise
        cond_topografia = ventana['Theta (rad)'].std() > 0.1
        
        if cond_combustible or cond_mf or cond_velocidad or cond_carga or cond_traccion or cond_topografia:
            return False
            
    elif vehiculo == "tracto":
        # Heavily calibrated tripwires for large diesel engine tractocamión
        cond_combustible = (ventana['Fuel flow (L/s)'] > 0.05).any()
        cond_mf = (ventana['mf'] > 80.0).any()
        cond_velocidad = ventana['Speed_Smooth'].mean() < 3.0
        cond_carga = ventana['Carga (kg)'].mean() < 1000.0  # limit set to 1 ton
        cond_traccion = (aceleracion_aprox >= -0.1).mean() < 0.60
        cond_topografia = ventana['Theta (rad)'].std() > 0.1
        
        if cond_combustible or cond_mf or cond_velocidad or cond_carga or cond_traccion or cond_topografia:
            return False
            
    return True

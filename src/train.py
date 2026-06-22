import torch
import numpy as np
import gc
from models import SuspensionPINN, ThermodynamicPINN
from physics import calculate_suspension_residual, calculate_required_power, calculate_available_power
from data_utils import evaluate_tripwire

def train_suspension_pinn(df, m_base, c_damping, k1, k2=None, gravity=9.81, 
                          window_size=100, epochs_init=1500, epochs_normal=150, 
                          epochs_jump=1000, lr_net=1e-3, lr_mass=10.0, 
                          optimize_k=False, lr_k=5000.0):
    """
    Trains the Suspension PINN.
    Uses the incremental state approach (model and optimizer instantiated outside the window loop).
    """
    print("Initializing suspension PINN...")
    model = SuspensionPINN(m_base, optimize_k=optimize_k, k_init=130000.0, initial_extra=0.0)
    
    params = [{'params': model.net.parameters(), 'lr': lr_net}]
    params.append({'params': model.gross_mass, 'lr': lr_mass})
    if optimize_k:
        params.append({'params': model.k_base, 'lr': lr_k})
        
    optimizer = torch.optim.Adam(params)
    
    payload_results = []
    times = []
    real_payloads = []
    last_time = None
    
    print("Training suspension PINN...")
    for idx, i in enumerate(range(0, len(df) - window_size, window_size)):
        window = df.iloc[i:i + window_size]
        
        # Local time scaling
        t_local = window['Acum time (s)'].values - window['Acum time (s)'].values[0]
        t_tensor = torch.tensor(t_local, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        x_real_tensor = torch.tensor(window['Deflection_m'].values, dtype=torch.float32).view(-1, 1)
        
        current_start_time = window['Acum time (s)'].iloc[0]
        
        # Determine training epochs based on temporal continuity
        if idx == 0:
            epochs = epochs_init
        else:
            if last_time is not None and (current_start_time - last_time) > 5.0:
                epochs = epochs_jump
            else:
                epochs = epochs_normal
                
        # Gradient descent loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            x_pred = model(t_tensor)
            
            # Loss Data (mm scale to keep errors readable)
            data_loss = torch.mean(((x_pred - x_real_tensor) * 1000.0) ** 2)
            
            # Autograd differentiation
            v_grad = torch.autograd.grad(x_pred, t_tensor, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
            a_grad = torch.autograd.grad(v_grad, t_tensor, grad_outputs=torch.ones_like(v_grad), create_graph=True)[0]
            
            # Loss Physics
            m_act = model.get_mass()
            k_desc = model.get_k() if optimize_k else None
            k_val = k_desc if optimize_k else k1
            
            residual_kN = calculate_suspension_residual(x_pred, v_grad, a_grad, m_act, m_base, c_damping, k_val, k2, gravity)
            physics_loss = torch.mean(residual_kN) ** 2
            
            total_loss = data_loss + physics_loss
            total_loss.backward()
            optimizer.step()
            
        last_time = window['Acum time (s)'].iloc[-1]
        payload = model.get_mass().item() - m_base
        
        payload_results.append(payload)
        times.append(last_time)
        real_payloads.append(window['Carga (kg)'].mean())
        
        if (i // window_size) % 5 == 0:
            print(f"Window {i//window_size} | Time {last_time:.0f}s | PINN: {payload:.1f} kg | Real: {real_payloads[-1]:.1f} kg")
            
    print("Training complete.")
    return np.array(times), np.array(real_payloads), np.array(payload_results)


def train_thermo_pinn(df, m_base, m_max=2500.0, eta=0.12, lhv_f=43000000.0, 
                      rho_f=0.750, c_d=0.35, a_f=2.5, f_r=0.01, rho_air=1.17, 
                      gravity=9.81, idle_burn=0.0, vehicle="peugeot", 
                      window_size=60, epochs_init=3000, epochs_normal=1500, 
                      lr_net=1e-3, lr_mass=10.0, use_mask=False, use_memory=True, 
                      warmup_epochs=300, physics_weight=10.0, use_huber=True):
    """
    Trains the Thermodynamic (OBD) PINN.
    Re-instantiates the model/optimizer for each window, with optional parameter inheritance (memory).
    """
    print(f"Initializing thermodynamic PINN for {vehicle}...")
    
    payload_results = []
    times = []
    real_payloads = []
    
    memory_mass = m_base + 0.0
    
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i + window_size].copy()
        
        # Apply Cable Trampa filters
        if not evaluate_tripwire(window, vehicle):
            continue
            
        model = ThermodynamicPINN(m_base, m_max)
        
        if use_memory:
            with torch.no_grad():
                model.gross_mass.fill_(memory_mass)
                
        optimizer = torch.optim.Adam([
            {'params': model.net.parameters(), 'lr': lr_net},
            {'params': model.gross_mass, 'lr': lr_mass}
        ])
        
        t_local = window['Acum time (s)'].values - window['Acum time (s)'].values[0]
        t_tensor = torch.tensor(t_local, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        v_real_tensor = torch.tensor(window['Speed_Smooth'].values, dtype=torch.float32).view(-1, 1)
        flow_tensor = torch.tensor(window['Fuel_L_s_real'].values, dtype=torch.float32).view(-1, 1)
        theta_tensor = torch.tensor(window['Theta (rad)'].values, dtype=torch.float32).view(-1, 1)
        mf_tensor = torch.tensor(window['mf'].values, dtype=torch.float32).view(-1, 1)
        
        mask_tensor = None
        if use_mask and 'Mask' in window.columns:
            mask_tensor = torch.tensor(window['Mask'].values, dtype=torch.float32).view(-1, 1)
            
        epochs = epochs_init if len(payload_results) == 0 else epochs_normal
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            v_pred = model(t_tensor)
            data_loss = torch.mean((v_pred - v_real_tensor) ** 2)
            
            # Autograd for acceleration
            v_grad = torch.autograd.grad(v_pred, t_tensor, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
            
            m_act = model.get_mass()
            
            # Physics power calculations
            P_req = calculate_required_power(v_real_tensor, v_grad, theta_tensor, mf_tensor, m_act, c_d, a_f, f_r, rho_air, gravity)
            P_disp = calculate_available_power(flow_tensor, eta, lhv_f, rho_f, idle_burn)
            
            # Calculate physical residual loss
            if use_huber:
                physics_err = torch.nn.functional.huber_loss(P_disp, P_req, delta=1.0, reduction='none')
            else:
                physics_err = (P_disp - P_req) ** 2
                
            if mask_tensor is not None:
                physics_loss = torch.mean(physics_err * mask_tensor)
            else:
                physics_loss = torch.mean(physics_err)
                
            total_loss = data_loss + (physics_weight * physics_loss)
            
            # Regularize base load
            tare_loss = torch.relu(m_base - m_act) ** 2
            total_loss = total_loss + (20.0 * tare_loss)
            
            total_loss.backward()
            
            # Warm-up schedule (freeze mass gradients in early epochs)
            if epoch < warmup_epochs:
                if model.gross_mass.grad is not None:
                    model.gross_mass.grad.zero_()
                    
            optimizer.step()
            
        discovered_mass = model.get_mass().item()
        
        if use_memory:
            if m_max is not None:
                memory_mass = float(np.clip(discovered_mass, m_base, m_max))
            else:
                memory_mass = max(discovered_mass, m_base)
        else:
            memory_mass = discovered_mass
            
        payload = memory_mass - m_base
        
        payload_results.append(payload)
        times.append(window['Acum time (s)'].iloc[-1])
        real_payloads.append(window['Carga (kg)'].mean())
        
        print(f"Window | Time {times[-1]:.0f}s | PINN: {payload:.1f} kg | Real: {real_payloads[-1]:.1f} kg")
        
        # Prevent PyTorch graph memory leaks
        del model, optimizer, t_tensor, v_real_tensor, flow_tensor, theta_tensor, mf_tensor
        if mask_tensor is not None:
            del mask_tensor
        gc.collect()
        
    print("Training complete.")
    return np.array(times), np.array(real_payloads), np.array(payload_results)

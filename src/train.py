import torch
import numpy as np
import gc
from models import SuspensionPINN, TermodinamicaPINN
from physics import calcular_residual_suspension, calcular_potencia_requerida, calcular_potencia_disponible
from data_utils import evaluar_cable_trampa

def entrenar_suspension_pinn(df, m_base, c_amort, k1, k2=None, gravedad=9.81, 
                             tam_ventana=100, epocas_init=1500, epocas_normal=150, 
                             epocas_jump=1000, lr_net=1e-3, lr_masa=10.0, 
                             optimize_k=False, lr_k=5000.0):
    """
    Trains the Suspension PINN.
    Uses the incremental state approach (model and optimizer instantiated outside the window loop).
    """
    print("Initializing Suspension PINN...")
    modelo = SuspensionPINN(m_base, optimize_k=optimize_k, k_init=130000.0, initial_extra=0.0)
    
    params = [{'params': modelo.red.parameters(), 'lr': lr_net}]
    params.append({'params': modelo.masa_bruta, 'lr': lr_masa})
    if optimize_k:
        params.append({'params': modelo.k_base, 'lr': lr_k})
        
    optimizador = torch.optim.Adam(params)
    
    resultados_carga = []
    tiempos = []
    cargas_reales = []
    ultimo_tiempo = None
    
    print("Starting training loop across time windows...")
    for idx, i in enumerate(range(0, len(df) - tam_ventana, tam_ventana)):
        ventana = df.iloc[i:i + tam_ventana]
        
        # Local time scaling
        t_local = ventana['Acum time (s)'].values - ventana['Acum time (s)'].values[0]
        t_tensor = torch.tensor(t_local, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        x_real_tensor = torch.tensor(ventana['Deflexion_m'].values, dtype=torch.float32).view(-1, 1)
        
        tiempo_inicio_actual = ventana['Acum time (s)'].iloc[0]
        
        # Determine training epochs based on temporal continuity
        if idx == 0:
            epocas = epocas_init
        else:
            if ultimo_tiempo is not None and (tiempo_inicio_actual - ultimo_tiempo) > 5.0:
                epocas = epocas_jump
            else:
                epocas = epocas_normal
                
        # Gradient descent loop
        for epoca in range(epocas):
            optimizador.zero_grad()
            
            x_pred = modelo(t_tensor)
            
            # Loss Data (mm scale to keep errors readable)
            loss_datos = torch.mean(((x_pred - x_real_tensor) * 1000.0) ** 2)
            
            # Autograd differentiation
            v_grad = torch.autograd.grad(x_pred, t_tensor, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
            a_grad = torch.autograd.grad(v_grad, t_tensor, grad_outputs=torch.ones_like(v_grad), create_graph=True)[0]
            
            # Loss Physics
            m_act = modelo.get_masa()
            k_desc = modelo.get_k() if optimize_k else None
            k_val = k_desc if optimize_k else k1
            
            residual_kN = calcular_residual_suspension(x_pred, v_grad, a_grad, m_act, m_base, c_amort, k_val, k2, gravedad)
            loss_fisica = torch.mean(residual_kN) ** 2
            
            loss_total = loss_datos + loss_fisica
            loss_total.backward()
            optimizador.step()
            
        ultimo_tiempo = ventana['Acum time (s)'].iloc[-1]
        carga_util = modelo.get_masa().item() - m_base
        
        resultados_carga.append(carga_util)
        tiempos.append(ultimo_tiempo)
        cargas_reales.append(ventana['Carga (kg)'].mean())
        
        if (i // tam_ventana) % 5 == 0:
            print(f"Window {i//tam_ventana} | Time {ultimo_tiempo:.0f}s | PINN: {carga_util:.1f} kg | Real: {cargas_reales[-1]:.1f} kg")
            
    print("Processing Completed!")
    return np.array(tiempos), np.array(cargas_reales), np.array(resultados_carga)


def entrenar_termo_pinn(df, m_base, m_max=2500.0, eta=0.12, lhv_f=43000000.0, 
                        rho_f=0.750, c_d=0.35, a_f=2.5, f_r=0.01, rho_air=1.17, 
                        gravedad=9.81, idle_burn=0.0, vehiculo="peugeot", 
                        tam_ventana=60, epocas_init=3000, epocas_normal=1500, 
                        lr_net=1e-3, lr_masa=10.0, use_mask=False, use_memory=True, 
                        warmup_epochs=300, peso_fisica=10.0, usar_huber=True):
    """
    Trains the Thermodynamic (OBD) PINN.
    Re-instantiates the model/optimizer for each window, with optional parameter inheritance (memory).
    """
    print(f"Initializing Thermodynamic PINN for {vehiculo}...")
    
    resultados_carga = []
    tiempos = []
    cargas_reales = []
    
    masa_memoria = m_base + 0.0
    
    for i in range(0, len(df) - tam_ventana, tam_ventana):
        ventana = df.iloc[i:i + tam_ventana].copy()
        
        # Apply Cable Trampa filters
        if not evaluar_cable_trampa(ventana, vehiculo):
            continue
            
        modelo = TermodinamicaPINN(m_base, m_max)
        
        if use_memory:
            with torch.no_grad():
                modelo.masa_bruta.fill_(masa_memoria)
                
        optimizador = torch.optim.Adam([
            {'params': modelo.red.parameters(), 'lr': lr_net},
            {'params': modelo.masa_bruta, 'lr': lr_masa}
        ])
        
        t_local = ventana['Acum time (s)'].values - ventana['Acum time (s)'].values[0]
        t_tensor = torch.tensor(t_local, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        v_real_tensor = torch.tensor(ventana['Speed_Smooth'].values, dtype=torch.float32).view(-1, 1)
        flujo_tensor = torch.tensor(ventana['Fuel_L_s_real'].values, dtype=torch.float32).view(-1, 1)
        theta_tensor = torch.tensor(ventana['Theta (rad)'].values, dtype=torch.float32).view(-1, 1)
        mf_tensor = torch.tensor(ventana['mf'].values, dtype=torch.float32).view(-1, 1)
        
        mascara_tensor = None
        if use_mask and 'Mascara' in ventana.columns:
            mascara_tensor = torch.tensor(ventana['Mascara'].values, dtype=torch.float32).view(-1, 1)
            
        epocas = epocas_init if len(resultados_carga) == 0 else epocas_normal
        
        for epoca in range(epocas):
            optimizador.zero_grad()
            
            v_pred = modelo(t_tensor)
            loss_datos = torch.mean((v_pred - v_real_tensor) ** 2)
            
            # Autograd for acceleration
            v_grad = torch.autograd.grad(v_pred, t_tensor, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
            
            m_act = modelo.get_masa()
            
            # Physics power calculations
            P_req = calcular_potencia_requerida(v_real_tensor, v_grad, theta_tensor, mf_tensor, m_act, c_d, a_f, f_r, rho_air, gravedad)
            P_disp = calcular_potencia_disponible(flujo_tensor, eta, lhv_f, rho_f, idle_burn)
            
            # Calculate physical residual loss
            if usar_huber:
                err_fisico = torch.nn.functional.huber_loss(P_disp, P_req, delta=1.0, reduction='none')
            else:
                err_fisico = (P_disp - P_req) ** 2
                
            if mascara_tensor is not None:
                loss_fisica = torch.mean(err_fisico * mascara_tensor)
            else:
                loss_fisica = torch.mean(err_fisico)
                
            loss_total = loss_datos + (peso_fisica * loss_fisica)
            
            # Regularize base load
            loss_tara = torch.relu(m_base - m_act) ** 2
            loss_total = loss_total + (20.0 * loss_tara)
            
            loss_total.backward()
            
            # Warm-up schedule (freeze mass gradients in early epochs)
            if epoca < warmup_epochs:
                if modelo.masa_bruta.grad is not None:
                    modelo.masa_bruta.grad.zero_()
                    
            optimizador.step()
            
        masa_descubierta = modelo.get_masa().item()
        
        if use_memory:
            if m_max is not None:
                masa_memoria = float(np.clip(masa_descubierta, m_base, m_max))
            else:
                masa_memoria = max(masa_descubierta, m_base)
        else:
            masa_memoria = masa_descubierta
            
        carga_util = masa_memoria - m_base
        
        resultados_carga.append(carga_util)
        tiempos.append(ventana['Acum time (s)'].iloc[-1])
        cargas_reales.append(ventana['Carga (kg)'].mean())
        
        print(f"Valid Window | Time {tiempos[-1]:.0f}s | PINN: {carga_util:.1f} kg | Real: {cargas_reales[-1]:.1f} kg")
        
        # Prevent PyTorch graph memory leaks
        del modelo, optimizador, t_tensor, v_real_tensor, flujo_tensor, theta_tensor, mf_tensor
        if mascara_tensor is not None:
            del mascara_tensor
        gc.collect()
        
    print("Processing Completed!")
    return np.array(tiempos), np.array(cargas_reales), np.array(resultados_carga)

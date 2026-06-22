import torch

def calcular_potencia_requerida(v_real, v_grad, theta, mf, m_actual, c_d, a_f, f_r, rho_air, gravedad=9.81):
    """
    Computes required traction power (P_req_kW) in kilowatts based on longitudinal forces:
    F_req = F_drag + F_rolling + F_gravity + F_inertia
    """
    F_d = 0.5 * c_d * a_f * rho_air * (v_real ** 2)
    R_x = f_r * m_actual * gravedad * torch.cos(theta)
    F_g = m_actual * gravedad * torch.sin(theta)
    F_i = m_actual * mf * v_grad
    
    F_req = F_d + R_x + F_g + F_i
    
    # Restricts power calculation to positive traction work (bypassing braking periods)
    Potencia_Req_Watts = torch.relu(F_req * v_real)
    return Potencia_Req_Watts / 1000.0


def calcular_potencia_disponible(flujo_combustible, eta, lhv_f, rho_f, idle_burn=0.0):
    """
    Computes available chemical fuel power (P_disp_kW) in kilowatts:
    P_disp = rho_fuel * (fuel_flow - idle_burn) * LHV * efficiency
    """
    flujo_traccion = torch.relu(flujo_combustible - idle_burn)
    Potencia_Disp_Watts = rho_f * flujo_traccion * lhv_f * eta
    return Potencia_Disp_Watts / 1000.0


def calcular_residual_suspension(x_pred, v, a, m_actual, m_base, c_amort, k1, k2=None, gravedad=9.81):
    """
    Computes the residual of the vertical suspension dynamic equation:
    residual = M*a + C*v + F_spring(x) - (M - M_base)*g
    Returns residual in KiloNewtons (kN) to balance loss scale with data loss.
    """
    peso_carga = (m_actual - m_base) * gravedad
    
    if k2 is not None:
        fuerza_resorte = (k1 * x_pred) + (k2 * (x_pred ** 2))
    else:
        fuerza_resorte = k1 * x_pred
        
    residual_N = (m_actual * a) + (c_amort * v) + fuerza_resorte - peso_carga
    return residual_N / 1000.0

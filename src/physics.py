import torch

def calculate_required_power(v_real, v_grad, theta, mf, m_current, c_d, a_f, f_r, rho_air, gravity=9.81):
    """
    Computes required traction power (P_req_kW) in kilowatts based on longitudinal forces:
    F_req = F_drag + F_rolling + F_gravity + F_inertia
    """
    F_d = 0.5 * c_d * a_f * rho_air * (v_real ** 2)
    R_x = f_r * m_current * gravity * torch.cos(theta)
    F_g = m_current * gravity * torch.sin(theta)
    F_i = m_current * mf * v_grad
    
    F_req = F_d + R_x + F_g + F_i
    
    # Restricts power calculation to positive traction work (bypassing braking periods)
    power_req_watts = torch.relu(F_req * v_real)
    return power_req_watts / 1000.0


def calculate_available_power(fuel_flow, eta, lhv_f, rho_f, idle_burn=0.0):
    """
    Computes available chemical fuel power (P_disp_kW) in kilowatts:
    P_disp = rho_fuel * (fuel_flow - idle_burn) * LHV * efficiency
    """
    traction_flow = torch.relu(fuel_flow - idle_burn)
    power_disp_watts = rho_f * traction_flow * lhv_f * eta
    return power_disp_watts / 1000.0


def calculate_suspension_residual(x_pred, v, a, m_current, m_base, c_damping, k1, k2=None, gravity=9.81):
    """
    Computes the residual of the vertical suspension dynamic equation:
    residual = M*a + C*v + F_spring(x) - (M - M_base)*g
    Returns residual in KiloNewtons (kN) to balance loss scale with data loss.
    """
    payload_weight = (m_current - m_base) * gravity
    
    if k2 is not None:
        spring_force = (k1 * x_pred) + (k2 * (x_pred ** 2))
    else:
        spring_force = k1 * x_pred
        
    residual_n = (m_current * a) + (c_damping * v) + spring_force - payload_weight
    return residual_n / 1000.0

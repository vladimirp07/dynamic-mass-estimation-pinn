import torch
import torch.nn as nn

class TermodinamicaPINN(nn.Module):
    """
    PINN for thermodynamic mass estimation based on power balance equations.
    Reconstructs vehicle velocity v(t) while optimizing payload mass as a free parameter.
    """
    def __init__(self, m_base, m_max=2500.0):
        super(TermodinamicaPINN, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.m_base = m_base
        self.m_max = m_max
        self.masa_bruta = nn.Parameter(torch.tensor([m_base + 0.0], requires_grad=True))

    def get_masa(self):
        # Clamps optimized mass to realistic physical boundaries
        if self.m_max is not None:
            return torch.clamp(self.masa_bruta, min=self.m_base, max=self.m_max)
        return torch.clamp(self.masa_bruta, min=self.m_base)

    def forward(self, t):
        return self.red(t)


class SuspensionPINN(nn.Module):
    """
    PINN for mechanical mass estimation using suspension deflection data.
    Reconstructs suspension vertical deflection x(t) while optimizing payload mass,
    and optionally calibrating the spring stiffness coefficient K.
    """
    def __init__(self, m_base, optimize_k=False, k_init=130000.0, initial_extra=0.0):
        super(SuspensionPINN, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.m_base = m_base
        self.optimize_k = optimize_k
        self.masa_bruta = nn.Parameter(torch.tensor([m_base + initial_extra], requires_grad=True))
        
        if optimize_k:
            self.k_base = nn.Parameter(torch.tensor([k_init], requires_grad=True))

    def get_masa(self):
        # Optimized mass cannot be lower than the vehicle tare weight
        return torch.clamp(self.masa_bruta, min=self.m_base)

    def get_k(self):
        if self.optimize_k:
            # Clamps stiffness to a realistic physical domain
            return torch.clamp(self.k_base, min=90000.0, max=200000.0)
        return None

    def forward(self, t):
        return self.red(t)

import torch
import torch.nn as nn

class ThermodynamicPINN(nn.Module):
    """
    PINN for thermodynamic mass estimation based on power balance equations.
    Reconstructs vehicle velocity v(t) while optimizing payload mass as a free parameter.
    """
    def __init__(self, m_base, m_max=2500.0):
        super(ThermodynamicPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.m_base = m_base
        self.m_max = m_max
        self.gross_mass = nn.Parameter(torch.tensor([m_base + 0.0], requires_grad=True))

    def get_mass(self):
        # Clamps optimized mass to realistic physical boundaries
        if self.m_max is not None:
            return torch.clamp(self.gross_mass, min=self.m_base, max=self.m_max)
        return torch.clamp(self.gross_mass, min=self.m_base)

    def forward(self, t):
        return self.net(t)


class SuspensionPINN(nn.Module):
    """
    PINN for mechanical mass estimation using suspension deflection data.
    Reconstructs suspension vertical deflection x(t) while optimizing payload mass,
    and optionally calibrating the spring stiffness coefficient K.
    """
    def __init__(self, m_base, optimize_k=False, k_init=130000.0, initial_extra=0.0):
        super(SuspensionPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.m_base = m_base
        self.optimize_k = optimize_k
        self.gross_mass = nn.Parameter(torch.tensor([m_base + initial_extra], requires_grad=True))
        
        if optimize_k:
            self.k_base = nn.Parameter(torch.tensor([k_init], requires_grad=True))

    def get_mass(self):
        # Optimized mass cannot be lower than the vehicle tare weight
        return torch.clamp(self.gross_mass, min=self.m_base)

    def get_k(self):
        if self.optimize_k:
            # Clamps stiffness to a realistic physical domain
            return torch.clamp(self.k_base, min=90000.0, max=200000.0)
        return None

    def forward(self, t):
        return self.net(t)

"""
Core implementations of chaotic maps and dynamical systems.

This module provides clean, educational implementations of classic chaotic systems
including the logistic map, Henon map, and Lorenz attractor.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DynamicalSystem:
    """Base class for dynamical systems with metadata."""
    name: str
    dimension: int
    parameters: Dict[str, float]
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None
    description: str = ""


class LogisticMap(DynamicalSystem):
    """
    The logistic map: x_{n+1} = r * x_n * (1 - x_n)
    
    A simple 1D map that exhibits period-doubling route to chaos.
    """
    
    def __init__(self, r: float = 3.8):
        super().__init__(
            name="Logistic Map",
            dimension=1,
            parameters={"r": r},
            bounds=((0, 1),),
            description="Classic 1D map showing period-doubling bifurcation"
        )
        self.r = r
    
    def __call__(self, x: float) -> float:
        """Apply one iteration of the logistic map."""
        return self.r * x * (1 - x)
    
    def iterate(self, x0: float, n: int) -> np.ndarray:
        """Generate n iterations starting from x0."""
        trajectory = np.zeros(n)
        trajectory[0] = x0
        
        for i in range(1, n):
            trajectory[i] = self(trajectory[i-1])
        
        return trajectory
    
    def bifurcation_data(self, r_range: Tuple[float, float] = (2.5, 4.0), 
                        n_r: int = 1000, n_discard: int = 200, n_plot: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bifurcation diagram data."""
        r_values = np.linspace(r_range[0], r_range[1], n_r)
        r_plot = []
        x_plot = []
        
        for r in r_values:
            self.r = r
            x = 0.5  # Initial condition
            
            # Discard transient
            for _ in range(n_discard):
                x = self(x)
            
            # Collect data
            for _ in range(n_plot):
                x = self(x)
                r_plot.append(r)
                x_plot.append(x)
        
        return np.array(r_plot), np.array(x_plot)


class HenonMap(DynamicalSystem):
    """
    The Henon map: 
    x_{n+1} = 1 - a * x_n^2 + y_n
    y_{n+1} = b * x_n
    
    A 2D invertible map with strange attractor.
    """
    
    def __init__(self, a: float = 1.4, b: float = 0.3):
        super().__init__(
            name="Henon Map",
            dimension=2,
            parameters={"a": a, "b": b},
            bounds=((-3, 3), (-3, 3)),
            description="2D map with fractal strange attractor"
        )
        self.a = a
        self.b = b
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply one iteration of the Henon map."""
        x, y = state
        x_new = 1 - self.a * x**2 + y
        y_new = self.b * x
        return np.array([x_new, y_new])
    
    def iterate(self, x0: np.ndarray, n: int) -> np.ndarray:
        """Generate n iterations starting from x0."""
        trajectory = np.zeros((n, 2))
        trajectory[0] = x0
        
        for i in range(1, n):
            trajectory[i] = self(trajectory[i-1])
        
        return trajectory
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix at given state."""
        x, y = state
        return np.array([
            [-2 * self.a * x, 1],
            [self.b, 0]
        ])


class LorenzSystem(DynamicalSystem):
    """
    The Lorenz system:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    
    The iconic butterfly attractor.
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        super().__init__(
            name="Lorenz System",
            dimension=3,
            parameters={"sigma": sigma, "rho": rho, "beta": beta},
            bounds=((-30, 30), (-30, 30), (0, 60)),
            description="The butterfly attractor - first strange attractor discovered"
        )
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def derivatives(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """Compute derivatives for ODE integration."""
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])
    
    def integrate(self, x0: np.ndarray, t_span: Tuple[float, float], 
                  n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the system using RK4."""
        from scipy.integrate import solve_ivp
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.derivatives, t_span, x0, t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y.T
    
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix at given state."""
        x, y, z = state
        return np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])


class RosslerSystem(DynamicalSystem):
    """
    The Rössler system:
    dx/dt = -y - z
    dy/dt = x + a * y
    dz/dt = b + z * (x - c)
    
    A simpler strange attractor with a single spiral.
    """
    
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        super().__init__(
            name="Rössler System",
            dimension=3,
            parameters={"a": a, "b": b, "c": c},
            bounds=((-20, 20), (-20, 20), (0, 40)),
            description="Strange attractor with single spiral structure"
        )
        self.a = a
        self.b = b
        self.c = c
    
    def derivatives(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """Compute derivatives for ODE integration."""
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz])
    
    def integrate(self, x0: np.ndarray, t_span: Tuple[float, float], 
                  n_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the system using RK4."""
        from scipy.integrate import solve_ivp
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.derivatives, t_span, x0, t_eval=t_eval, method='RK45')
        
        return sol.t, sol.y.T


# Registry of available systems
SYSTEMS = {
    "logistic": LogisticMap,
    "henon": HenonMap,
    "lorenz": LorenzSystem,
    "rossler": RosslerSystem
}


def get_system(name: str, **kwargs) -> DynamicalSystem:
    """Factory function to create dynamical systems by name."""
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system: {name}. Available: {list(SYSTEMS.keys())}")
    
    return SYSTEMS[name](**kwargs)
"""
Visualization utilities for chaotic systems.

This module provides reusable plotting functions for trajectories,
attractors, bifurcation diagrams, and phase space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Tuple, List, Callable, Union
import warnings

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_time_series(trajectory: np.ndarray, 
                     title: str = "Time Series",
                     xlabel: str = "Time",
                     ylabel: str = "Value",
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """Plot time series data for 1D or multi-dimensional systems."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if trajectory.ndim == 1:
        ax.plot(trajectory, linewidth=0.5)
    else:
        for i in range(trajectory.shape[1]):
            ax.plot(trajectory[:, i], linewidth=0.5, label=f"Dimension {i+1}")
        ax.legend()
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    return fig


def plot_phase_space_2d(trajectory: np.ndarray,
                        title: str = "Phase Space",
                        xlabel: str = "x",
                        ylabel: str = "y",
                        figsize: Tuple[int, int] = (8, 8),
                        color_by_time: bool = True) -> plt.Figure:
    """Plot 2D phase space trajectory."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if trajectory.shape[1] < 2:
        raise ValueError("Need at least 2D data for phase space plot")
    
    if color_by_time:
        points = ax.scatter(trajectory[:, 0], trajectory[:, 1], 
                           c=np.arange(len(trajectory)), 
                           cmap='viridis', s=0.5, alpha=0.6)
        plt.colorbar(points, ax=ax, label='Time')
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5, alpha=0.8)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'ro', markersize=8, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=8, label='End')
        ax.legend()
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_attractor_3d(trajectory: np.ndarray,
                      title: str = "3D Attractor",
                      figsize: Tuple[int, int] = (12, 10),
                      elev: float = 20,
                      azim: float = 45,
                      color_by_time: bool = True) -> plt.Figure:
    """Plot 3D attractor with customizable viewing angle."""
    if trajectory.shape[1] < 3:
        raise ValueError("Need 3D data for 3D attractor plot")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by_time:
        points = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                           c=np.arange(len(trajectory)), cmap='plasma', 
                           s=0.1, alpha=0.6)
        plt.colorbar(points, ax=ax, label='Time', pad=0.1)
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                linewidth=0.5, alpha=0.8)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.view_init(elev=elev, azim=azim)
    
    return fig


def plot_bifurcation(r_values: np.ndarray, 
                     x_values: np.ndarray,
                     title: str = "Bifurcation Diagram",
                     xlabel: str = "Parameter r",
                     ylabel: str = "x",
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot bifurcation diagram for 1D maps."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(r_values, x_values, ',k', markersize=0.5, alpha=0.25)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(r_values.min(), r_values.max())
    
    return fig


def plot_return_map(trajectory: np.ndarray,
                    title: str = "Return Map",
                    figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """Plot return map (x_n vs x_{n+1}) for 1D systems."""
    if trajectory.ndim > 1:
        trajectory = trajectory[:, 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(trajectory[:-1], trajectory[1:], alpha=0.5, s=1)
    ax.plot([trajectory.min(), trajectory.max()], 
            [trajectory.min(), trajectory.max()], 
            'r--', alpha=0.5, label='y=x')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('$x_n$', fontsize=12)
    ax.set_ylabel('$x_{n+1}$', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    
    return fig


def animate_trajectory(trajectory: np.ndarray,
                      title: str = "Animated Trajectory",
                      interval: int = 50,
                      trail_length: int = 100,
                      figsize: Tuple[int, int] = (10, 8)) -> FuncAnimation:
    """Animate the evolution of a trajectory."""
    if trajectory.shape[1] == 2:
        return _animate_2d(trajectory, title, interval, trail_length, figsize)
    elif trajectory.shape[1] == 3:
        return _animate_3d(trajectory, title, interval, trail_length, figsize)
    else:
        raise ValueError("Animation only supports 2D and 3D trajectories")


def _animate_2d(trajectory: np.ndarray, title: str, interval: int, 
                trail_length: int, figsize: Tuple[int, int]) -> FuncAnimation:
    """Helper function for 2D animation."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_xlim(trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1)
    ax.set_ylim(trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    line, = ax.plot([], [], 'b-', linewidth=0.5)
    point, = ax.plot([], [], 'ro', markersize=8)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate(i):
        start_idx = max(0, i - trail_length)
        line.set_data(trajectory[start_idx:i+1, 0], 
                     trajectory[start_idx:i+1, 1])
        point.set_data([trajectory[i, 0]], [trajectory[i, 1]])
        return line, point
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(trajectory), interval=interval,
                        blit=True)
    
    return anim


def _animate_3d(trajectory: np.ndarray, title: str, interval: int,
                trail_length: int, figsize: Tuple[int, int]) -> FuncAnimation:
    """Helper function for 3D animation."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(trajectory[:, 0].min() - 1, trajectory[:, 0].max() + 1)
    ax.set_ylim(trajectory[:, 1].min() - 1, trajectory[:, 1].max() + 1)
    ax.set_zlim(trajectory[:, 2].min() - 1, trajectory[:, 2].max() + 1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    line, = ax.plot([], [], [], 'b-', linewidth=0.5)
    point, = ax.plot([], [], [], 'ro', markersize=8)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point
    
    def animate(i):
        start_idx = max(0, i - trail_length)
        line.set_data(trajectory[start_idx:i+1, 0], 
                     trajectory[start_idx:i+1, 1])
        line.set_3d_properties(trajectory[start_idx:i+1, 2])
        point.set_data([trajectory[i, 0]], [trajectory[i, 1]])
        point.set_3d_properties([trajectory[i, 2]])
        return line, point
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(trajectory), interval=interval,
                        blit=True)
    
    return anim


def plot_sensitivity(trajectories: List[np.ndarray],
                     labels: Optional[List[str]] = None,
                     title: str = "Sensitivity to Initial Conditions",
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot multiple trajectories to show sensitivity to initial conditions."""
    n_dims = trajectories[0].shape[1] if trajectories[0].ndim > 1 else 1
    
    if n_dims == 1:
        fig, ax = plt.subplots(figsize=figsize)
        for i, traj in enumerate(trajectories):
            label = labels[i] if labels else f"IC {i+1}"
            ax.plot(traj, linewidth=0.8, alpha=0.7, label=label)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
    else:
        fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
        if n_dims == 1:
            axes = [axes]
        
        for dim in range(n_dims):
            for i, traj in enumerate(trajectories):
                label = labels[i] if labels else f"IC {i+1}"
                axes[dim].plot(traj[:, dim], linewidth=0.8, alpha=0.7, label=label)
            axes[dim].set_ylabel(f"Dim {dim+1}", fontsize=12)
            if dim == 0:
                axes[dim].legend(loc='upper right')
        
        axes[-1].set_xlabel("Time", fontsize=12)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_poincare_section(trajectory: np.ndarray,
                         plane_normal: np.ndarray,
                         plane_point: np.ndarray,
                         title: str = "Poincaré Section",
                         figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """Plot Poincaré section for 3D systems."""
    if trajectory.shape[1] != 3:
        raise ValueError("Poincaré sections require 3D trajectories")
    
    # Find intersections with the plane
    intersections = []
    normal = plane_normal / np.linalg.norm(plane_normal)
    
    for i in range(len(trajectory) - 1):
        p1, p2 = trajectory[i], trajectory[i + 1]
        d1 = np.dot(p1 - plane_point, normal)
        d2 = np.dot(p2 - plane_point, normal)
        
        # Check if trajectory crosses plane
        if d1 * d2 < 0:
            # Linear interpolation to find intersection
            t = -d1 / (d2 - d1)
            intersection = p1 + t * (p2 - p1)
            intersections.append(intersection)
    
    if not intersections:
        warnings.warn("No intersections found with the specified plane")
        return plt.figure()
    
    intersections = np.array(intersections)
    
    # Project onto plane coordinates
    # Create orthonormal basis for the plane
    v1 = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    
    # Project intersections
    proj_x = [np.dot(p - plane_point, v1) for p in intersections]
    proj_y = [np.dot(p - plane_point, v2) for p in intersections]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(proj_x, proj_y, s=2, alpha=0.6)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Projection X", fontsize=12)
    ax.set_ylabel("Projection Y", fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    
    return fig


# CLI Demo functionality
def demo():
    """Run a demonstration of the visualization capabilities."""
    print("<  Strange Attractor Visualization Demo")
    print("=" * 40)
    
    from .maps import LogisticMap, LorenzSystem, HenonMap
    
    # Demo 1: Logistic Map Bifurcation
    print("\n1. Generating Logistic Map bifurcation diagram...")
    logistic = LogisticMap()
    r_vals, x_vals = logistic.bifurcation_data()
    fig1 = plot_bifurcation(r_vals, x_vals, 
                           title="Logistic Map: Route to Chaos")
    plt.show()
    
    # Demo 2: Lorenz Attractor
    print("\n2. Generating Lorenz Attractor...")
    lorenz = LorenzSystem()
    t, trajectory = lorenz.integrate(np.array([1, 1, 1]), (0, 50), n_points=5000)
    fig2 = plot_attractor_3d(trajectory, title="The Lorenz Butterfly")
    plt.show()
    
    # Demo 3: Henon Map
    print("\n3. Generating Henon Map...")
    henon = HenonMap()
    trajectory = henon.iterate(np.array([0, 0]), 10000)
    fig3 = plot_phase_space_2d(trajectory[1000:], title="Henon Strange Attractor",
                              color_by_time=False)
    plt.show()
    
    print("\n( Demo complete! Explore more in the notebooks/")


if __name__ == "__main__":
    demo()
#!/usr/bin/env python3
"""
Create an icon for the Strange Attractor Math Course app.
Generates a Lorenz attractor visualization as the icon.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import os
from pathlib import Path

def lorenz_system(state, sigma=10, rho=28, beta=8/3):
    """Lorenz system derivatives."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def integrate_lorenz(x0, t_span, dt=0.01):
    """Simple RK4 integration of Lorenz system."""
    t0, tf = t_span
    t = np.arange(t0, tf, dt)
    n = len(t)
    
    trajectory = np.zeros((n, 3))
    trajectory[0] = x0
    
    for i in range(1, n):
        # RK4 integration
        k1 = lorenz_system(trajectory[i-1])
        k2 = lorenz_system(trajectory[i-1] + dt*k1/2)
        k3 = lorenz_system(trajectory[i-1] + dt*k2/2)
        k4 = lorenz_system(trajectory[i-1] + dt*k3)
        
        trajectory[i] = trajectory[i-1] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return trajectory

def create_icon(size=1024):
    """Create icon image of Lorenz attractor."""
    print(f"Creating {size}x{size} icon...")
    
    # Generate Lorenz attractor
    x0 = np.array([1, 1, 1])
    trajectory = integrate_lorenz(x0, (0, 30), dt=0.005)
    
    # Skip transient
    trajectory = trajectory[1000:]
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 10), facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('none')
    
    # Plot attractor with gradient colors
    points = trajectory
    colors = plt.cm.plasma(np.linspace(0, 1, len(points)))
    
    # Plot as a continuous line with color gradient
    for i in range(len(points)-1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2],
                color=colors[i], linewidth=2, alpha=0.8)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Remove axes and labels for clean icon
    ax.set_axis_off()
    
    # Adjust limits to center the attractor
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([0, 50])
    
    plt.tight_layout()
    
    # Save to high-res image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the RGBA buffer
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    image = Image.frombytes("RGBA", (w, h), buf)
    
    # Resize to target size
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    # Add circular mask for macOS style
    mask = Image.new('L', (size, size), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    
    # Create output with circular mask
    output = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    output.paste(image, mask=mask)
    
    plt.close()
    
    return output

def create_icns():
    """Create .icns file for macOS."""
    print("Creating icon set for macOS...")
    
    # Create resources directory
    resources_dir = Path(__file__).parent / "resources"
    resources_dir.mkdir(exist_ok=True)
    
    # Icon sizes needed for macOS
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # Create temporary iconset directory
    iconset_dir = resources_dir / "StrangeAttractor.iconset"
    iconset_dir.mkdir(exist_ok=True)
    
    # Generate icon at largest size
    base_icon = create_icon(1024)
    
    # Save as PNG first
    png_path = resources_dir / "icon.png"
    base_icon.save(png_path, "PNG")
    print(f"Created PNG icon: {png_path}")
    
    # Create all required sizes
    for size in sizes:
        # Normal resolution
        icon = base_icon.resize((size, size), Image.Resampling.LANCZOS)
        icon.save(iconset_dir / f"icon_{size}x{size}.png")
        
        # Retina resolution (2x)
        if size <= 512:  # Max retina is 1024x1024
            icon_2x = base_icon.resize((size*2, size*2), Image.Resampling.LANCZOS)
            icon_2x.save(iconset_dir / f"icon_{size}x{size}@2x.png")
    
    # Convert to .icns using iconutil
    icns_path = resources_dir / "StrangeAttractor.icns"
    os.system(f"iconutil -c icns -o '{icns_path}' '{iconset_dir}'")
    
    # Clean up iconset directory
    os.system(f"rm -rf '{iconset_dir}'")
    
    print(f"Created ICNS icon: {icns_path}")
    
    return png_path, icns_path

if __name__ == "__main__":
    png_path, icns_path = create_icns()
    print("\nIcon creation complete!")
    print(f"PNG icon: {png_path}")
    print(f"ICNS icon: {icns_path}")
#!/usr/bin/env python3
"""
Strange Attractor Math Course Launcher

This script provides a graphical launcher for the course materials.
"""

import sys
import os
import subprocess
import webbrowser
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:
    print("Installing tkinter is required. On macOS, it should be included with Python.")
    print("If missing, try: brew install python-tk")
    sys.exit(1)


class StrangeAttractorLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Strange Attractor Math Course")
        self.root.geometry("600x500")
        
        # Set icon if available
        icon_path = Path(__file__).parent / "resources" / "icon.png"
        if icon_path.exists():
            try:
                photo = tk.PhotoImage(file=str(icon_path))
                root.wm_iconphoto(False, photo)
            except:
                pass
        
        # Style
        style = ttk.Style()
        style.theme_use('default')
        
        # Header
        header_frame = tk.Frame(root, bg='#2C3E50', height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🌀 Strange Attractor Math Course",
            font=('Helvetica', 24, 'bold'),
            fg='white',
            bg='#2C3E50'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            header_frame,
            text="From PEMDAS to Chaos Theory",
            font=('Helvetica', 12),
            fg='#ECF0F1',
            bg='#2C3E50'
        )
        subtitle_label.pack()
        
        # Main content
        main_frame = tk.Frame(root, bg='#ECF0F1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Buttons
        button_style = {
            'font': ('Helvetica', 14),
            'width': 30,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 2,
            'cursor': 'hand2'
        }
        
        # Launch Jupyter Lab
        jupyter_btn = tk.Button(
            main_frame,
            text="📓 Launch Interactive Notebooks",
            command=self.launch_jupyter,
            bg='#3498DB',
            fg='white',
            activebackground='#2980B9',
            **button_style
        )
        jupyter_btn.pack(pady=10)
        
        # Run Demo
        demo_btn = tk.Button(
            main_frame,
            text="🎨 Run Visualization Demo",
            command=self.run_demo,
            bg='#E74C3C',
            fg='white',
            activebackground='#C0392B',
            **button_style
        )
        demo_btn.pack(pady=10)
        
        # Show Cheatsheet
        cheat_btn = tk.Button(
            main_frame,
            text="📋 View Math Cheatsheet",
            command=self.show_cheatsheet,
            bg='#2ECC71',
            fg='white',
            activebackground='#27AE60',
            **button_style
        )
        cheat_btn.pack(pady=10)
        
        # Open README
        readme_btn = tk.Button(
            main_frame,
            text="📖 Read Documentation",
            command=self.open_readme,
            bg='#9B59B6',
            fg='white',
            activebackground='#8E44AD',
            **button_style
        )
        readme_btn.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to explore chaos!")
        status_bar = tk.Label(
            root,
            textvariable=self.status_var,
            bg='#34495E',
            fg='white',
            anchor=tk.W,
            padx=10
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def update_status(self, message):
        """Update status bar message."""
        self.status_var.set(message)
        self.root.update()
        
    def launch_jupyter(self):
        """Launch Jupyter Lab with the notebooks."""
        self.update_status("Starting Jupyter Lab...")
        
        try:
            # Change to project directory
            os.chdir(Path(__file__).parent)
            
            # Start Jupyter Lab
            subprocess.Popen([
                sys.executable, '-m', 'jupyter', 'lab',
                '--notebook-dir=notebooks'
            ])
            
            self.update_status("Jupyter Lab launched! Check your browser.")
            
            # Show info dialog
            messagebox.showinfo(
                "Jupyter Lab Started",
                "Jupyter Lab is starting...\n\n"
                "It should open automatically in your browser.\n"
                "If not, navigate to: http://localhost:8888\n\n"
                "Start with 00_logistic_map.ipynb!"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Jupyter Lab:\n{e}")
            self.update_status("Error launching Jupyter Lab")
    
    def run_demo(self):
        """Run the visualization demo."""
        self.update_status("Running visualization demo...")
        
        try:
            # Create a new terminal window to run the demo
            applescript = '''
            tell application "Terminal"
                activate
                do script "cd '{}' && {} -m src.visualise --demo"
            end tell
            '''.format(Path(__file__).parent, sys.executable)
            
            subprocess.run(['osascript', '-e', applescript])
            
            self.update_status("Demo launched in Terminal!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run demo:\n{e}")
            self.update_status("Error running demo")
    
    def show_cheatsheet(self):
        """Display the mathematical cheatsheet."""
        self.update_status("Showing cheatsheet...")
        
        try:
            # Create a new terminal window for the cheatsheet
            applescript = '''
            tell application "Terminal"
                activate
                do script "cd '{}' && {} -m src.cheatsheet"
            end tell
            '''.format(Path(__file__).parent, sys.executable)
            
            subprocess.run(['osascript', '-e', applescript])
            
            self.update_status("Cheatsheet displayed in Terminal!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show cheatsheet:\n{e}")
            self.update_status("Error showing cheatsheet")
    
    def open_readme(self):
        """Open README in browser."""
        self.update_status("Opening documentation...")
        
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            webbrowser.open(f"file://{readme_path}")
            self.update_status("Documentation opened in browser!")
        else:
            messagebox.showerror("Error", "README.md not found!")
            self.update_status("README not found")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = StrangeAttractorLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
# THIS IS AI

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Color mappings for tile types
TILE_COLORS = {
    0: (0, 105, 148),  # Water - deep blue
    1: (200, 230, 255),  # Ice - light blue
    2: (237, 201, 175),  # Desert - sandy
    3: (124, 252, 0),  # Grassland - green
    4: (34, 139, 34),  # Forest - dark green
    5: (128, 128, 128),  # City - gray
}


class WeatherSimulatorViewer:
    def __init__(self, simulator, root=None):
        """
        Initialize the weather simulator viewer.

        Args:
            simulator: WeatherSimulator instance
            root: Optional tkinter root window (creates one if not provided)
        """
        self.simulator = simulator
        self.grid = simulator.grid
        self.generation = 0

        # History tracking
        self.temp_history = []
        self.pollution_history = []

        # Create or use provided root window
        if root is None:
            self.root = tk.Tk()
            self.owns_root = True
        else:
            self.root = root
            self.owns_root = False

        self.root.title("Weather Simulator")

        # Configure grid weight for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._setup_ui()
        self._update_display()

    def _setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Left panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Simulation View", padding="5")
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Canvas for grid display
        canvas_size = 600
        self.canvas = tk.Canvas(viz_frame, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Right panel - Controls and Stats
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Statistics display
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)

        ttk.Label(stats_frame, text="Generation:").grid(row=0, column=0, sticky=tk.W)
        self.gen_label = ttk.Label(stats_frame, text="0", font=('Arial', 10, 'bold'))
        self.gen_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(stats_frame, text="Avg Temperature:").grid(row=1, column=0, sticky=tk.W)
        self.temp_label = ttk.Label(stats_frame, text="0.0", font=('Arial', 10, 'bold'))
        self.temp_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(stats_frame, text="Avg Pollution:").grid(row=2, column=0, sticky=tk.W)
        self.pollution_label = ttk.Label(stats_frame, text="0.0", font=('Arial', 10, 'bold'))
        self.pollution_label.grid(row=2, column=1, sticky=tk.W, padx=5)

        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Controls", padding="10")
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Next Step", command=self._next_step).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Next Year (365 steps)", command=self._next_year).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset History", command=self._reset_history).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Show Correlation Plot", command=self._show_correlation_plot).pack(fill=tk.X,
                                                                                                         pady=2)

        # Legend
        legend_frame = ttk.LabelFrame(control_frame, text="Tile Legend", padding="10")
        legend_frame.pack(fill=tk.X, pady=5)

        tile_names = ["Water", "Ice", "Desert", "Grassland", "Forest", "City"]
        for i, name in enumerate(tile_names):
            color = TILE_COLORS[i]
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'

            frame = ttk.Frame(legend_frame)
            frame.pack(fill=tk.X, pady=2)

            canvas = tk.Canvas(frame, width=20, height=20, bg=hex_color, highlightthickness=1)
            canvas.pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text=name).pack(side=tk.LEFT)

        # Info label
        info_frame = ttk.LabelFrame(control_frame, text="Info", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.info_text = tk.Text(info_frame, height=8, width=30, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.insert('1.0',
                              "Hover over tiles to see details.\n\nColors represent tile types.\nBrightness indicates temperature.\nOverlay shows pollution levels.")
        self.info_text.config(state=tk.DISABLED)

        # Bind mouse motion for hover info
        self.canvas.bind('<Motion>', self._on_hover)

    def _create_visualization_image(self):
        """Create a PIL Image of the current grid state."""
        h, w = self.grid.tiles.shape

        # Create base image from tile types
        img_array = np.zeros((h, w, 3), dtype=np.uint8)
        for tile_type, color in TILE_COLORS.items():
            mask = self.grid.tiles == tile_type
            img_array[mask] = color

        # Modulate brightness based on temperature (normalized)
        temp_norm = self.grid.temps.copy()
        temp_min, temp_max = temp_norm.min(), temp_norm.max()
        if temp_max > temp_min:
            temp_norm = (temp_norm - temp_min) / (temp_max - temp_min)
        else:
            temp_norm = np.ones_like(temp_norm) * 0.5

        # Apply temperature as brightness adjustment
        brightness_factor = 0.5 + temp_norm * 0.5  # Range from 0.5 to 1.0
        img_array = (img_array * brightness_factor[:, :, np.newaxis]).astype(np.uint8)

        # Add pollution overlay (reddish tint)
        pollution_norm = self.grid.pollution.copy()
        poll_max = pollution_norm.max()
        if poll_max > 0:
            pollution_norm = pollution_norm / poll_max

            # Add red channel for pollution
            red_overlay = (pollution_norm * 255 * 0.3).astype(np.uint8)
            img_array[:, :, 0] = np.minimum(255, img_array[:, :, 0] + red_overlay)

        # Create PIL Image
        img = Image.fromarray(img_array, 'RGB')

        # Resize to canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            img = img.resize((canvas_width, canvas_height), Image.NEAREST)
        else:
            img = img.resize((600, 600), Image.NEAREST)

        return img

    def _update_display(self):
        """Update the visualization display."""
        # Create and display image
        img = self._create_visualization_image()
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Update statistics
        avg_temp = float(np.mean(self.grid.temps))
        avg_pollution = float(np.mean(self.grid.pollution))

        self.gen_label.config(text=str(self.generation))
        self.temp_label.config(text=f"{avg_temp:.2f}")
        self.pollution_label.config(text=f"{avg_pollution:.2f}")

        # Store history
        self.temp_history.append(avg_temp)
        self.pollution_history.append(avg_pollution)

    def _next_step(self):
        """Advance simulation by one step."""
        self.simulator.next()
        self.generation += 1
        self._update_display()

    def _next_year(self):
        """Advance simulation by 365 steps."""
        for _ in range(365):
            self.simulator.next()
            self.generation += 1
            self._update_display()

    def _reset_history(self):
        """Reset the history tracking."""
        self.temp_history = []
        self.pollution_history = []
        self.generation = 0
        self._update_display()

    def _show_correlation_plot(self):
        """Show a plot of temperature vs pollution correlation."""
        if len(self.temp_history) < 2:
            tk.messagebox.showinfo("Info", "Not enough data. Run simulation first.")
            return

        # Create new window for plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Temperature vs Pollution Correlation")

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

        # Time series plot
        generations = list(range(len(self.temp_history)))
        ax1.plot(generations, self.temp_history, label='Avg Temperature', color='red', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(generations, self.pollution_history, label='Avg Pollution', color='gray', alpha=0.7)

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Average Temperature', color='red')
        ax1_twin.set_ylabel('Average Pollution', color='gray')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        ax1.set_title('Temperature and Pollution Over Time')
        ax1.grid(True, alpha=0.3)

        norm_temps = np.array(self.temp_history).astype(np.float32)
        tmp_std_dev = np.std(norm_temps)
        min_temp = np.min(norm_temps)
        max_temp = np.max(norm_temps)
        norm_temps = (norm_temps - min_temp) / (max_temp - min_temp)

        norm_pols = np.array(self.pollution_history).astype(np.float32)
        pol_std_dev = np.std(norm_pols)
        min_pol = np.min(norm_pols)
        max_pol = np.max(norm_pols)
        norm_pols = (norm_pols - min_pol) / (max_pol - min_pol)
        ax2.plot(generations, norm_temps, label='Avg Temperature', color='red', alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(generations, norm_pols, label='Avg Pollution', color='gray', alpha=0.7)

        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Temperature', color='red')
        ax2_twin.set_ylabel('Average Pollution', color='gray')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2_twin.tick_params(axis='y', labelcolor='gray')
        ax2.set_title(f'Normalized Temperature (std - {tmp_std_dev:.2f}) and Pollution (std - {pol_std_dev:.2f}) Over Time')
        ax2.grid(True, alpha=0.3)

        # Scatter plot - correlation
        ax3.scatter(self.pollution_history, self.temp_history, alpha=0.5, s=10)
        ax3.set_xlabel('Average Pollution')
        ax3.set_ylabel('Average Temperature')
        ax3.set_title('Temperature vs Pollution Correlation')
        ax3.grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(self.temp_history) > 1:
            corr = np.corrcoef(self.pollution_history, self.temp_history)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add close button
        ttk.Button(plot_window, text="Close", command=plot_window.destroy).pack(pady=5)

    def _on_hover(self, event):
        """Display tile information on hover."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        h, w = self.grid.tiles.shape

        # Convert canvas coordinates to grid coordinates
        if canvas_width > 1 and canvas_height > 1:
            row = int(event.y * h / canvas_height)
            col = int(event.x * w / canvas_width)

            if 0 <= row < h and 0 <= col < w:
                tile_type = int(self.grid.tiles[row, col])
                temp = float(self.grid.temps[row, col])
                pollution = float(self.grid.pollution[row, col])
                wind = self.grid.winds[row, col]

                tile_names = ["Water", "Ice", "Desert", "Grassland", "Forest", "City"]
                tile_name = tile_names[tile_type] if tile_type < len(tile_names) else "Unknown"

                info = f"Position: ({row}, {col})\n"
                info += f"Type: {tile_name}\n"
                info += f"Temperature: {temp:.2f}\n"
                info += f"Pollution: {pollution:.2f}\n"
                info += f"Wind: ({wind[0]}, {wind[1]})"

                self.info_text.config(state=tk.NORMAL)
                self.info_text.delete('1.0', tk.END)
                self.info_text.insert('1.0', info)
                self.info_text.config(state=tk.DISABLED)

    def run(self):
        """Start the tkinter main loop (only if this viewer created the root)."""
        if self.owns_root:
            self.root.mainloop()

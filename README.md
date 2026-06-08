# Digital History Tracker & Dashboard

A futuristic, high-tech, and beautiful activity logger and visualizer built on top of OpenRecall. 

This repository contains a heavily customized, sci-fi "50,000-year quantum" inspired dashboard layout featuring:
- **Interactive Three.js 3D Quantum Logo** with custom particle effects and interactive controls.
- **Glassmorphic Sci-Fi Dashboard UI** with animations, cosmic backgrounds, and clean modular layout.
- **Self-contained Workspace** so the custom styling is safe from global package overwrites.

## Project Structure
- `run_openrecall.py`: The entry point script to run the local OpenRecall server.
- `init_openrecall_db.py`: Initializes the SQLite database.
- `openrecall/`: Local package directory containing customized Flask logic (`app.py`) and frontend templates (`templates/index.html`).
- `quantum_logo_50k (1).html`: Independent prototype showing the premium Three.js 3D logo animation.

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install Flask sqlite3 python-doctr
   ```
   *(Note: Make sure Pywin32 is installed on Windows if you run into screen capture issues)*

2. **Initialize Database** (First time only):
   ```bash
   python init_openrecall_db.py
   ```

3. **Start the Dashboard**:
   ```bash
   python run_openrecall.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

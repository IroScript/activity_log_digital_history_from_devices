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
- `linux_setup/`: Contains scripts and GNOME extensions to enable automated background screenshots on Ubuntu Wayland.

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

## Ubuntu / Linux Wayland Autostart Setup

On modern Ubuntu (Wayland), taking automated screenshots in the background is blocked by default for security reasons.

To set up the tracker to run automatically on boot and bypass the Wayland screenshot restrictions, run the included setup script:

```bash
chmod +x linux_setup/install.sh
./linux_setup/install.sh
```

**Important**: After running the script:
1. You **must** Log Out and Log In again for GNOME to load the extension.
2. Open the **Extensions** app in Ubuntu and ensure **Unsafe Mode Autostart** is turned **ON**.

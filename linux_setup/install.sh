#!/bin/bash
set -e

echo "Setting up OpenRecall for Linux (Wayland/GNOME) Autostart..."

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Install GNOME Extension for Unsafe Mode (to allow screenshots on Wayland)
EXT_DIR="$HOME/.local/share/gnome-shell/extensions/unsafe-mode-autostart@openrecall"
echo "Installing GNOME Extension to: $EXT_DIR"
mkdir -p "$EXT_DIR"
cp -r "$APP_DIR/linux_setup/unsafe-mode-autostart@openrecall/"* "$EXT_DIR/"

echo "Enabling GNOME Extension (Note: if you are on Wayland, you MUST log out and log back in for GNOME to detect it, then turn it on in the Extensions app)."
gnome-extensions enable unsafe-mode-autostart@openrecall || true

# 2. Setup Autostart Desktop Entry
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/openrecall.desktop"

echo "Creating autostart entry: $DESKTOP_FILE"
mkdir -p "$AUTOSTART_DIR"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Exec=/bin/bash -c "sleep 120 && cd $APP_DIR && $APP_DIR/venv/bin/python run_openrecall.py >> $APP_DIR/autostart.log 2>&1"
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=OpenRecall Tracker
Comment=Auto-start OpenRecall background tracker
EOF

chmod +x "$DESKTOP_FILE"

echo "================================================================"
echo "Setup Complete!"
echo "1. Please LOG OUT and LOG IN again to load the GNOME Extension."
echo "2. Open the 'Extensions' app on Ubuntu and make sure 'Unsafe Mode Autostart' is toggled ON."
echo "================================================================"

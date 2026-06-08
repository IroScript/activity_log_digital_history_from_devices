#!/usr/bin/env python3
"""
OpenRecall Launcher Script
This script runs the OpenRecall Flask application
"""

try:
    import threading
    import sys
    if sys.platform == "win32":
        try:
            import win32gui
            import win32process
            win32gui.GetWindowThreadProcessId = win32process.GetWindowThreadProcessId
        except ImportError:
            pass
            
    from openrecall.app import app, create_db, record_screenshot_thread
    
    print("=" * 60)
    print("Initializing Database...")
    create_db()
    print("Database initialized successfully.")
    print("=" * 60)
    
    print("Starting screenshot recording thread...")
    t = threading.Thread(target=record_screenshot_thread, daemon=True)
    t.start()
    print("Screenshot recording thread started.")
    print("=" * 60)
    
    print("Starting OpenRecall...")
    print("Access the application at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
    
except ImportError as e:
    print(f"Error importing OpenRecall: {e}")
    print("\nMissing dependencies. Please install:")
    print("  pip install python-doctr")
    print("\nNote: OpenRecall may have been designed for macOS and might not work fully on Windows.")
except Exception as e:
    print(f"Error running OpenRecall: {e}")
    import traceback
    traceback.print_exc()

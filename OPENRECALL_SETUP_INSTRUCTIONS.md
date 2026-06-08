# OpenRecall Setup Instructions for Windows

## Current Issue
OpenRecall is failing to start due to a PyTorch DLL initialization error on Windows.

**Error:** `[WinError 1114] A dynamic link library (DLL) initialization routine failed`

## Solutions

### Solution 1: Install Visual C++ Redistributables (RECOMMENDED)

PyTorch requires the Microsoft Visual C++ Redistributable. Download and install:

**Latest Visual C++ Redistributable:**
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Or visit: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

After installing, restart your computer and try running OpenRecall again.

### Solution 2: Reinstall PyTorch

Try reinstalling PyTorch to ensure all DLLs are properly registered:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Solution 3: Use CPU-only PyTorch

If you have GPU-related issues, try CPU-only version:

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## How to Run OpenRecall

Once the PyTorch issue is resolved, run:

```powershell
python run_openrecall.py
```

Or directly:

```powershell
python -c "from openrecall.app import app; app.run()"
```

## Known Limitations

1. **Windows Compatibility**: OpenRecall was likely designed for macOS (requires `pyobjc` which doesn't work on Windows)
2. **Version Conflicts**: OpenRecall has strict version requirements that conflict with newer packages
3. **Missing Entry Point**: OpenRecall doesn't have a proper CLI command configured

## Dependencies Status

✅ Installed:
- Flask, h5py, mss, rapidfuzz, shapely
- python-doctr (OCR functionality)
- sentence-transformers, torch, torchvision

❌ Not Available on Windows:
- pyobjc (macOS-only)

## Alternative: Check if OpenRecall Source is Available

If you have the OpenRecall source code, you could:
1. Fix the Windows compatibility issues
2. Add a proper entry point script
3. Update dependencies to use compatible versions

## Troubleshooting

If problems persist:
1. Check Python version (should be 3.12)
2. Ensure you're running PowerShell as Administrator
3. Try creating a fresh virtual environment
4. Consider using WSL2 (Windows Subsystem for Linux) for better compatibility

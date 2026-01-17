#!/usr/bin/env python3
"""
Wrapper to ensure msk2k_audio_qso_server_Q12.py can find msk2k_complete.py
"""
import sys
import os
from pathlib import Path

# Add the script's directory to Python path so imports work
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Log to a file for debugging
log_file = script_dir / "msk2k_server.log"
try:
    with open(log_file, "w") as f:
        f.write(f"Starting MSK2K server from {script_dir}\n")
        f.write(f"Python path: {sys.path}\n")
        f.flush()
        
        # Redirect stdout and stderr to the log file
        sys.stdout = f
        sys.stderr = f
        
        # Now import and run the actual server
        import msk2k_audio_qso_server_Q12
        
        f.write("About to call main()\n")
        f.flush()
        
        msk2k_audio_qso_server_Q12.main()
        
except Exception as e:
    with open(log_file, "a") as f:
        f.write(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc(file=f)

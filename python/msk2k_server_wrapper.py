#!/usr/bin/env python3
"""
Wrapper to ensure msk2k_audio_qso_server_Q12.py can find msk2k_complete.py
"""
import sys
from pathlib import Path

# Add the script's directory to Python path so imports work
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Print to stdout so Electron can see it
print(f"MSK2K Server starting from {script_dir}", flush=True)
print(f"Python path includes: {script_dir}", flush=True)

# Now import and run the actual server
import msk2k_audio_qso_server_Q12

print("Calling main()...", flush=True)
msk2k_audio_qso_server_Q12.main()

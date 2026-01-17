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

# Now import and run the actual server
import msk2k_audio_qso_server_Q12

if __name__ == "__main__":
    msk2k_audio_qso_server_Q12.main()

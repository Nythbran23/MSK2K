# MSK2K - Meteor Scatter Digital Mode

[![Build MSK2K](https://github.com/Nythbran23/MSK2K/actions/workflows/build.yml/badge.svg)](https://github.com/Nythbran23/MSK2K/actions/workflows/build.yml)

MSK2K is a cross-platform desktop application for meteor scatter communication using MSK modulation with the PSK2K protocol.

**MSK2K uses the same protocol as PSK2K but replaces BPSK modulation with MSK (Minimum Shift Keying):**
- Constant envelope (no zero crossings)
- ~2dB gain from full PA efficiency
- Continuous phase, cleaner spectrum
- 258-bit packets at 2000 baud (129ms)
- Rate 1/2, K=7 convolutional coding
- Soft-decision Viterbi with LLR accumulation

## Downloads

### Latest Release

Download the latest version for your platform:

- **macOS**: [MSK2K-x.x.x.dmg](https://github.com/Nythbran23/MSK2K/releases/latest) (Intel & Apple Silicon)
- **Windows**: [MSK2K-Setup-x.x.x.exe](https://github.com/Nythbran23/MSK2K/releases/latest)
- **Linux**: [MSK2K-x.x.x.AppImage](https://github.com/Nythbran23/MSK2K/releases/latest) or [.deb](https://github.com/Nythbran23/MSK2K/releases/latest)

## Features

- **Cross-platform**: Works on macOS, Windows, and Linux
- **Real-time audio I/O**: Full-duplex audio using PortAudio
- **MSK modulation**: Constant envelope for efficient amplifier operation
- **QSO automation**: AUTO-SEQ mode for hands-free operation
- **ADIF logging**: Automatic QSO logging in ADIF format
- **Web-based UI**: Modern, responsive interface
- **Multiple modes**: LISTEN, CQ, CALL, REPLY

## Installation

### macOS

1. Download `MSK2K-x.x.x.dmg`
2. Open the DMG file
3. Drag MSK2K to Applications folder
4. Right-click > Open (first time only to bypass Gatekeeper)
5. Grant microphone permission when prompted

### Windows

1. Download `MSK2K-Setup-x.x.x.exe`
2. Run the installer
3. Follow the installation wizard
4. Launch MSK2K from Start Menu

### Linux

**AppImage (Universal):**
```bash
chmod +x MSK2K-x.x.x.AppImage
./MSK2K-x.x.x.AppImage
```

**Debian/Ubuntu (.deb):**
```bash
sudo dpkg -i msk2k_x.x.x_amd64.deb
sudo apt-get install -f  # Install dependencies
msk2k
```

## Development

### Prerequisites

- Node.js 16+ and npm
- Python 3.9+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Nythbran23/MSK2K.git
cd MSK2K

# Install Node dependencies
npm install

# Install Python dependencies
pip3 install -r python/requirements.txt

# Run in development mode
npm start
```

### Building

```bash
# Build for current platform
npm run dist

# Build for specific platforms
npm run dist:mac     # macOS
npm run dist:win     # Windows
npm run dist:linux   # Linux

# Build for all platforms (requires appropriate OS or CI)
npm run dist:all
```

Build outputs will be in the `dist/` directory.

## Project Structure

```
MSK2K/
├── main.js                 # Electron main process
├── preload.js             # Electron preload script
├── package.json           # Node.js dependencies & build config
├── python/                # Python backend
│   ├── msk2k_audio_qso_server_Q12.py  # Main server
│   ├── msk2k_complete.py              # MSK2K encoder/decoder
│   └── requirements.txt               # Python dependencies
├── renderer/              # UI assets (served by Python)
│   └── msk2k_audio_qso_ui_Q12.html
├── build/                 # Build resources
│   ├── icon.icns         # macOS icon
│   ├── icon.ico          # Windows icon
│   ├── icon.png          # Linux icon
│   └── entitlements.mac.plist
└── .github/
    └── workflows/
        └── build.yml      # CI/CD for automated builds
```

## Usage

1. Launch MSK2K
2. Configure your audio devices in Settings
3. Enter your callsign
4. Select operating mode:
   - **LISTEN**: Receive only
   - **CQ**: Call CQ on your selected time slot
   - **CALL**: Call a specific station
   - **REPLY**: Reply to a received call
5. Enable AUTO-SEQ for automatic sequencing
6. Start operating!

## Audio Configuration

MSK2K supports separate RX and TX audio devices:

- **RX Device**: Select your audio input (radio receiver)
- **TX Device**: Select your audio output (radio transmitter)
- **Gain Controls**: Adjust RX and TX audio levels
- **Auto RX Gain**: Automatic level control for reception

### Testing with BlackHole (macOS)

For testing two instances on one Mac:

1. Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole)
2. Instance 1: TX to BlackHole 2ch
3. Instance 2: RX from BlackHole 2ch

## Technical Details

### Protocol Specifications

- **Modulation**: MSK (Minimum Shift Keying)
- **Baud Rate**: 2000 symbols/second
- **Packet Length**: 258 bits (129ms)
- **FEC**: Rate 1/2, K=7 convolutional code
- **Decoder**: Soft-decision Viterbi with LLR accumulation
- **Interleaving**: PSK2K-compatible
- **Audio Rate**: 48 kHz (default)

### Message Types

- **CQ**: `CQ <callsign> <locator>`
- **Reply**: `<callsign> <callsign> <report>`
- **Acknowledgment**: `<callsign> R<report>`
- **73**: `<callsign> 73`

### Report Codes

Signal quality reports (based on correlation percentage):
- **26**: Q ≥ 90% (Excellent)
- **27**: 80% ≤ Q < 90% (Very Good)
- **28**: 70% ≤ Q < 80% (Good)
- **29**: 60% ≤ Q < 70% (Fair)
- **21**: Q < 60% (Marginal)

## Troubleshooting

### Application won't start

- **macOS**: Check Console.app for errors, ensure Python 3 is installed
- **Windows**: Check Task Manager, ensure Python dependencies installed
- **Linux**: Run from terminal to see errors: `./MSK2K-x.x.x.AppImage`

### Port already in use

MSK2K uses port 8088 by default. If another application is using this port:
- Close other MSK2K instances
- Stop other applications using port 8088

### Audio devices not showing

- Grant microphone/audio permissions in system settings
- Restart the application after granting permissions
- Check that audio devices are properly connected

### Python backend fails

Ensure Python dependencies are installed:
```bash
pip3 install numpy scipy aiohttp sounddevice
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple platforms if possible
5. Submit a pull request

## License

Amateur radio use only.

## Author

**Roger Banks (GW4WND)**  
The DX Shop - https://thedxshop.com

Based on PSK2K by Klaus von der Heide (DJ5HG)

## Links

- **GitHub**: https://github.com/Nythbran23/MSK2K
- **The DX Shop**: https://thedxshop.com
- **Issues**: https://github.com/Nythbran23/MSK2K/issues

## Changelog

### v0.1.0 (Initial Release)
- Cross-platform Electron application
- MSK modulation with PSK2K protocol
- Real-time audio I/O
- QSO automation with AUTO-SEQ
- ADIF logging
- Support for macOS, Windows, and Linux

---

73 de GW4WND

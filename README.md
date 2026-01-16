# MSK2K - Meteor Scatter Digital Mode

[![Build MSK2K](https://github.com/Nythbran23/MSK2K/actions/workflows/build.yml/badge.svg)](https://github.com/Nythbran23/MSK2K/actions/workflows/build.yml)

**🎉 Zero Dependencies! No Python installation required!**

MSK2K is a self-contained desktop application for meteor scatter communication using MSK modulation with the PSK2K protocol. Everything is bundled - just download and run!

## ✨ Key Features

- **🚀 Zero Setup**: No Python, no pip, no dependencies to install
- **📦 Fully Self-Contained**: Python runtime bundled for each platform
- **🖥️ Cross-Platform**: macOS (Intel & Apple Silicon), Windows, Linux
- **📡 MSK Modulation**: Constant envelope for efficient amplifier operation (~2dB gain)
- **🎯 Real-time Audio**: Full-duplex audio processing
- **🤖 AUTO-SEQ Mode**: Hands-free QSO automation
- **📝 ADIF Logging**: Automatic QSO logging
- **🎨 Modern UI**: Responsive web-based interface

## 📥 Download & Install

### [📦 Latest Release →](https://github.com/Nythbran23/MSK2K/releases/latest)

Choose your platform:

### macOS

**Intel Macs:**
- Download: `MSK2K-x.x.x-x64.dmg`

**Apple Silicon (M1/M2/M3):**
- Download: `MSK2K-x.x.x-arm64.dmg`

**Installation:**
1. Open the DMG file
2. Drag MSK2K to Applications
3. First launch: Right-click > Open (to bypass Gatekeeper)
4. That's it! No other software needed.

### Windows

- Download: `MSK2K-Setup-x.x.x.exe`

**Installation:**
1. Run the installer
2. Follow the wizard
3. Launch from Start Menu
4. That's it! No Python installation needed.

### Linux

**AppImage (Universal):**
```bash
chmod +x MSK2K-x.x.x.AppImage
./MSK2K-x.x.x.AppImage
```

**Debian/Ubuntu (.deb):**
```bash
sudo dpkg -i msk2k_x.x.x_amd64.deb
msk2k
```

No Python installation required on any platform!

---

## 🚀 Quick Start

### First Launch

1. **MSK2K opens automatically** after installation
2. **Enter your callsign** - Top right corner, replace "NOCALL"
3. **Configure audio**:
   - Expand "Audio I/O" section
   - Select RX Device (radio's audio input)
   - Select TX Device (radio's audio output)
   - Click "Apply"
4. **Adjust levels** - Use TX/RX gain sliders in "Audio Levels"
5. **Start operating!**

### Operating Modes

- **LISTEN**: Receive only, decode all signals
- **CQ**: Call CQ on your selected time slot
- **CALL**: Call specific station (enter callsign)
- **REPLY**: Reply to received call (click callsign)

Enable **AUTO-SEQ** for automatic message sequencing!

### Signal Reports

Automatic reports based on decode quality:
- **26**: Excellent (≥90%)
- **27**: Very Good (80-90%)
- **28**: Good (70-80%)
- **29**: Fair (60-70%)
- **21**: Marginal (<60%)

---

## 🔧 Troubleshooting

### App Won't Open (macOS)

**Problem**: "MSK2K.app is damaged"

**Solution**:
```bash
xattr -cr /Applications/MSK2K.app
```
Then right-click > Open

### Port Already in Use

**Problem**: "Port 8088 is already in use"

**Solution**:
- Close other MSK2K instances
- Or restart your computer

### No Audio Devices

**macOS**: Grant microphone permission in System Preferences > Privacy
**Windows**: Enable "Allow desktop apps to access microphone" in Settings
**Linux**: Ensure user is in `audio` group: `sudo usermod -a -G audio $USER`

### Backend Error

If you see "Backend stopped unexpectedly", restart the app. If the problem persists, please file an issue on GitHub.

---

## 📡 Technical Specifications

- **Modulation**: MSK (Minimum Shift Keying)
- **Baud Rate**: 2000 symbols/second
- **Packet Length**: 258 bits (129ms)
- **FEC**: Rate 1/2, K=7 convolutional code
- **Decoder**: Soft-decision Viterbi with LLR accumulation
- **Audio Sample Rate**: 48 kHz
- **Bundled Python**: 3.11.x (self-contained, no installation needed)
- **Dependencies**: numpy 1.24.4, scipy 1.10.1, aiohttp 3.9.1, sounddevice 0.4.6

---

## 🛠️ Development

### Prerequisites

- Node.js 18+
- Python 3.9+ (for local development only)
- Git

### Setup

```bash
git clone https://github.com/Nythbran23/MSK2K.git
cd MSK2K
npm install
```

### Build Python Runtime

**macOS/Linux:**
```bash
chmod +x build-scripts/build-python-runtime.sh
./build-scripts/build-python-runtime.sh
```

**Windows:**
```cmd
build-scripts\build-python-runtime.bat
```

This creates `resources/python-runtime/<platform>/` with a complete Python venv.

### Run in Development

```bash
npm start
```

Development mode uses system Python. Production builds use bundled runtime.

### Build Distribution

```bash
npm run dist          # Current platform
npm run dist:mac      # macOS
npm run dist:win      # Windows
npm run dist:linux    # Linux
```

The bundled Python runtime will be included automatically.

---

## 🏗️ How It Works

### Bundled Python Runtime

MSK2K bundles a complete Python virtual environment for each platform:

```
MSK2K.app/
└── Contents/
    └── Resources/
        └── python-runtime/
            ├── mac-x64/      (Intel Mac)
            ├── mac-arm64/    (Apple Silicon)
            ├── win-x64/      (Windows)
            └── linux-x64/    (Linux)
```

On first launch:
1. Electron detects your platform
2. Extracts the correct Python runtime to `userData/python/`
3. Spawns the backend using the bundled Python
4. Opens the UI

No internet connection or external dependencies required!

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Test on multiple platforms if possible
4. Submit a pull request

### Building for All Platforms

GitHub Actions automatically builds for:
- macOS Intel (x64)
- macOS Apple Silicon (arm64)
- Windows (x64)
- Linux (x64)

Just push a tag and the builds happen automatically!

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

Amateur radio use encouraged!

---

## 👤 Author

**Roger Banks (GW4WND)**  
The DX Shop - https://thedxshop.com

Based on PSK2K by Klaus von der Heide (DJ5HG)

---

## 🔗 Links

- **GitHub**: https://github.com/Nythbran23/MSK2K
- **Issues**: https://github.com/Nythbran23/MSK2K/issues
- **Releases**: https://github.com/Nythbran23/MSK2K/releases
- **The DX Shop**: https://thedxshop.com

---

## 📝 Changelog

### v0.2.0-alpha - Zero Dependencies Release
- ✅ **Bundled Python runtime** - No external Python needed!
- ✅ **Multi-architecture support** - Intel & Apple Silicon for macOS
- ✅ **Pinned dependencies** - sounddevice 0.4.6 for Windows compatibility
- ✅ **Runtime extraction** - Automatic on first launch
- ✅ **NOCALL default** - Clean first-run experience
- ✅ **Improved error handling** - Better diagnostic messages

### v0.1.0-alpha - Initial Release
- Cross-platform Electron application
- MSK modulation with PSK2K protocol
- Real-time audio I/O
- QSO automation
- ADIF logging

---

## 💬 Support

- **Issues**: https://github.com/Nythbran23/MSK2K/issues
- **Discussions**: https://github.com/Nythbran23/MSK2K/discussions

---

**73 de GW4WND** 📻

*Enjoy meteor scatter with zero hassle!*

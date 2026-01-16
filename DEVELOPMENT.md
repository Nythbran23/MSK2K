# MSK2K Development Guide

This guide explains how to build and develop MSK2K with bundled Python runtimes.

## Architecture

MSK2K uses a hybrid approach:
- **Frontend**: Electron (Node.js + Chromium)
- **Backend**: Python (aiohttp server + signal processing)
- **Bundling**: Self-contained Python venv per platform

## Directory Structure

```
msk2k/
├── main.js                  # Electron main process
├── preload.js              # Electron preload script
├── package.json            # Node dependencies & build config
├── python/                 # Python backend source
│   ├── msk2k_audio_qso_server_Q12.py
│   ├── msk2k_complete.py
│   ├── msk2k_audio_qso_ui_Q12.html
│   └── requirements.txt    # Pinned Python dependencies
├── build-scripts/          # Build automation
│   ├── build-python-runtime.sh   # macOS/Linux venv builder
│   └── build-python-runtime.bat  # Windows venv builder
├── resources/             # Generated (gitignored)
│   └── python-runtime/
│       ├── mac-x64/       # Intel Mac venv
│       ├── mac-arm64/     # Apple Silicon venv
│       ├── win-x64/       # Windows venv
│       └── linux-x64/     # Linux venv
└── .github/workflows/
    └── build.yml          # CI/CD pipeline
```

## Development Setup

### Prerequisites

- Node.js 18+
- Python 3.9+
- Git

### Initial Setup

```bash
git clone https://github.com/Nythbran23/MSK2K.git
cd MSK2K
npm install
```

### Development Mode

In development, MSK2K uses your **system Python** directly (no bundling):

```bash
# Install Python dependencies
pip3 install -r python/requirements.txt

# Run the app
npm start
```

The app detects dev mode via `app.isPackaged === false` and spawns:
- macOS/Linux: `python3 python/msk2k_audio_qso_server_Q12.py`
- Windows: `python python/msk2k_audio_qso_server_Q12.py`

## Building Python Runtimes

Before building distributable packages, you need to create platform-specific Python runtimes.

### macOS

```bash
# Intel Mac
chmod +x build-scripts/build-python-runtime.sh
./build-scripts/build-python-runtime.sh

# This creates: resources/python-runtime/mac-x64/
```

For Apple Silicon, run on an M1/M2/M3 Mac:
```bash
./build-scripts/build-python-runtime.sh
# Creates: resources/python-runtime/mac-arm64/
```

### Windows

```cmd
build-scripts\build-python-runtime.bat
REM Creates: resources\python-runtime\win-x64\
```

### Linux

```bash
chmod +x build-scripts/build-python-runtime.sh
./build-scripts/build-python-runtime.sh
# Creates: resources/python-runtime/linux-x64/
```

## Building Distributable Packages

Once the Python runtime is built for your platform:

```bash
# Build for current platform
npm run dist

# Or specific platform
npm run dist:mac      # macOS DMG + ZIP
npm run dist:win      # Windows NSIS installer + portable
npm run dist:linux    # Linux AppImage + DEB
```

Output: `dist/` directory with installers.

## CI/CD Pipeline

GitHub Actions automatically builds for all platforms when you push a tag:

```bash
git tag -a v0.2.0-alpha -m "Release v0.2.0-alpha"
git push origin v0.2.0-alpha
```

The workflow (`.github/workflows/build.yml`):
1. Runs on 4 separate runners (macOS-Intel, macOS-ARM, Windows, Linux)
2. Each builds its own Python runtime
3. Packages with electron-builder
4. Creates a GitHub Release with all artifacts

### Supported Build Targets

- **macOS Intel**: macos-13 runner (x64)
- **macOS Apple Silicon**: macos-14 runner (arm64)
- **Windows**: windows-latest (x64)
- **Linux**: ubuntu-latest (x64)

## Python Runtime Details

### What Gets Bundled

Each platform runtime includes:
- Python 3.11 interpreter
- Virtual environment (venv)
- Pinned packages:
  - numpy==1.24.4
  - scipy==1.10.1
  - aiohttp==3.9.1
  - sounddevice==0.4.6
- Application files:
  - msk2k_audio_qso_server_Q12.py
  - msk2k_complete.py
  - msk2k_audio_qso_ui_Q12.html
- Version marker: `runtime_version.txt`

### Runtime Extraction Flow

On first launch:

1. **Platform Detection**: `main.js` detects OS and architecture
2. **Version Check**: Compares `userData/python/runtime_version.txt` with bundled version
3. **Extraction**: If needed, copies `resources/python-runtime/<platform>/` → `userData/python/`
4. **Spawning**: Runs `userData/python/bin/python3` (or `python.exe` on Windows)

On subsequent launches, extraction is skipped if versions match.

### UserData Locations

Python runtime is extracted to:
- **macOS**: `~/Library/Application Support/MSK2K/python/`
- **Windows**: `%APPDATA%\MSK2K\python\`
- **Linux**: `~/.config/MSK2K/python/`

## Updating Python Dependencies

Edit `python/requirements.txt`:

```
numpy==1.24.4
scipy==1.10.1
aiohttp==3.9.1
sounddevice==0.4.6
```

**Important**: Pin exact versions for reproducible builds!

After updating:
1. Rebuild Python runtimes for all platforms
2. Bump `runtime_version.txt` (e.g., v0.2.0 → v0.2.1)
3. Test on each platform

## Testing

### Local Testing

```bash
# Development mode (uses system Python)
npm start

# Production build (uses bundled runtime)
./build-scripts/build-python-runtime.sh
npm run dist
open dist/MSK2K.app   # macOS
```

### Testing Runtime Extraction

Delete the extracted runtime to force re-extraction:

```bash
# macOS
rm -rf ~/Library/Application\ Support/MSK2K/python/

# Then launch the app
```

Check logs for extraction messages.

### Multi-Platform Testing

Use virtual machines or GitHub Actions to test all platforms:
- Windows: VirtualBox or Parallels
- Linux: VirtualBox or Docker
- macOS: Native or GitHub Actions

## Debugging

### Enable Verbose Logging

The backend prints to stdout/stderr, which Electron captures:

```javascript
pythonProcess.stdout.on('data', (data) => {
  console.log(`[Backend]: ${data}`);
});
```

Check:
- macOS: Console.app
- Windows: Command prompt (run from terminal)
- Linux: Terminal output

### Common Issues

**"Backend failed to start"**
- Check Python runtime was built: `ls resources/python-runtime/`
- Verify packages installed: `<runtime>/bin/pip3 list`

**"Port already in use"**
- Kill existing instance: `lsof -ti:8088 | xargs kill -9` (macOS/Linux)

**Audio device errors (Windows)**
- Ensure sounddevice==0.4.6 (not 0.5.x)

## Release Checklist

Before releasing:

- [ ] Update version in `package.json`
- [ ] Update `runtime_version.txt` in build scripts
- [ ] Test on macOS (Intel + Apple Silicon)
- [ ] Test on Windows
- [ ] Test on Linux
- [ ] Update CHANGELOG in README.md
- [ ] Create and push tag
- [ ] Wait for CI/CD to complete
- [ ] Test release artifacts
- [ ] Publish release notes

## Performance

### Bundle Sizes

Approximate sizes per platform:
- macOS: ~150MB (DMG includes Python + packages)
- Windows: ~120MB (NSIS installer)
- Linux: ~140MB (AppImage)

Compressed downloads are significantly smaller.

### Startup Time

- First launch: ~2-3 seconds (runtime extraction)
- Subsequent launches: <1 second (no extraction)

## Security

### Code Signing (macOS)

For distribution outside development:
1. Get Apple Developer account
2. Create signing certificate
3. Configure in electron-builder:
   ```json
   "mac": {
     "identity": "Developer ID Application: Your Name"
   }
   ```

### Windows Signing

For production:
1. Get code signing certificate
2. Configure in electron-builder:
   ```json
   "win": {
     "certificateFile": "path/to/cert.pfx",
     "certificatePassword": "password"
   }
   ```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key principles:
- Pin dependencies for reproducibility
- Test on multiple platforms
- Keep build scripts simple
- Document breaking changes

---

Questions? Open an issue: https://github.com/Nythbran23/MSK2K/issues

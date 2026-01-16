# MSK2K Development Setup

Quick guide for developers working on MSK2K.

## Initial Setup

```bash
# Clone the repo
git clone https://github.com/Nythbran23/MSK2K.git
cd MSK2K

# Install Node dependencies
npm install

# Install Python dependencies
pip3 install -r python/requirements.txt
```

## Running in Development

```bash
npm start
```

This will:
1. Start the Python backend on port 8088
2. Launch Electron
3. Open DevTools automatically
4. Show Python console output

The application will reload when you make changes to `main.js` or `preload.js`.
Reload the window (Cmd+R / Ctrl+R) to see changes in the UI.

## Building

### Build for current platform
```bash
npm run dist
```

Output: `dist/` folder with installers

### Build for specific platforms

```bash
npm run dist:mac      # macOS (DMG + ZIP)
npm run dist:win      # Windows (NSIS installer + portable)
npm run dist:linux    # Linux (AppImage + DEB)
```

### Build for all platforms (macOS only)

```bash
npm run dist:all
```

Note: Cross-platform builds work best from macOS. For best results, build on each platform natively or use CI/CD.

## Directory Structure

```
MSK2K/
├── main.js                 # Electron main process (Node.js)
│                          # - Spawns Python backend
│                          # - Creates browser window
│                          # - Handles app lifecycle
│
├── preload.js             # Security bridge
│                          # - Exposes safe APIs to renderer
│
├── package.json           # Node/Electron config
│                          # - Dependencies
│                          # - Build configuration
│
├── python/                # Python backend
│   ├── msk2k_audio_qso_server_Q12.py
│   ├── msk2k_complete.py
│   └── requirements.txt
│
├── renderer/              # Frontend assets
│   └── msk2k_audio_qso_ui_Q12.html
│
└── build/                 # Build resources
    ├── icon.icns         # macOS
    ├── icon.ico          # Windows
    └── icon.png          # Linux
```

## Development Workflow

### Making Changes

1. **Python Backend** (`python/*.py`):
   - Edit files
   - Restart the app (quit and `npm start`)

2. **UI** (`renderer/*.html`):
   - Edit files
   - Reload window (Cmd+R / Ctrl+R)

3. **Electron** (`main.js`, `preload.js`):
   - Edit files
   - Restart the app (quit and `npm start`)

### Testing on Multiple Platforms

**macOS**:
- Native development environment
- Can build for all platforms

**Windows**:
- Use Windows machine or VM
- Build Windows installers natively

**Linux**:
- Use Linux machine or VM
- Build AppImage and DEB natively

**Or use CI/CD**: Push to GitHub, let Actions build all platforms!

## Python Backend Details

The Python backend:
- Runs as a subprocess spawned by Electron
- Serves HTTP on `localhost:8088`
- Provides WebSocket for real-time updates
- Uses PortAudio via sounddevice for audio I/O

### Python Dependencies

- `numpy`: DSP and signal processing
- `scipy`: Filters, FFT, signal analysis
- `aiohttp`: Async HTTP server and WebSocket
- `sounddevice`: PortAudio wrapper for audio I/O

### Adding Python Dependencies

1. Add to `python/requirements.txt`
2. Install: `pip3 install -r python/requirements.txt`
3. Test: `npm start`
4. Rebuild: `npm run dist`

## Electron Builder Configuration

See `package.json` → `build` section for:
- App ID and metadata
- File inclusion patterns
- Platform-specific settings
- Icon paths
- Code signing (when configured)

### Icons

Place icons in `build/`:
- `icon.icns` - macOS (1024x1024 PNG → icns)
- `icon.ico` - Windows (256x256 PNG → ico)
- `icon.png` - Linux (512x512 or 1024x1024 PNG)

Tools for icon creation:
- macOS: `iconutil` (built-in)
- Windows: ImageMagick, GIMP
- Linux: Any PNG editor

## Debugging

### Electron Main Process
```bash
# Run with Node inspector
electron --inspect=5858 .
```

Then open Chrome to `chrome://inspect`

### Python Backend

Check console output when running `npm start`. Python stdout/stderr is logged.

To run Python backend standalone:
```bash
cd python
python3 msk2k_audio_qso_server_Q12.py --host 127.0.0.1 --port 8088
```

### UI / Renderer Process

DevTools automatically open in development mode.

## CI/CD with GitHub Actions

The project includes `.github/workflows/build.yml` which:

1. Triggers on:
   - Push to `main`
   - Pull requests
   - Version tags (`v*`)
   - Manual workflow dispatch

2. Builds for:
   - macOS (Intel + Apple Silicon)
   - Windows (64-bit)
   - Linux (x64)

3. Uploads artifacts:
   - macOS: DMG + ZIP
   - Windows: NSIS installer
   - Linux: AppImage + DEB

4. Creates releases:
   - Automatically when you push a tag like `v0.1.0`
   - Attaches all platform builds to the release

### Creating a Release

```bash
# Tag the release
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

GitHub Actions will automatically build and publish the release!

## Common Issues

### Port 8088 already in use
- Kill existing MSK2K processes
- Or change PORT in `main.js`

### Python not found
- Ensure Python 3.9+ is installed
- On Windows, add to PATH
- On macOS, use `python3` command

### Build fails on macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Ensure Node.js 16+ is installed

### Build fails on Windows
- Install Windows Build Tools: `npm install --global windows-build-tools`
- Or install Visual Studio with C++ tools

### Build fails on Linux
- Install dependencies: `sudo apt-get install -y libgtk-3-dev libasound2-dev`

## Performance Tips

- Python backend is single-threaded (asyncio)
- Audio processing happens in separate thread
- WebSocket updates are throttled to 10 Hz
- Large ADIF logs loaded asynchronously

## Security Considerations

- `contextIsolation: true` - Renderer isolated from Node
- `nodeIntegration: false` - No Node in renderer
- `preload.js` - Controlled API exposure only
- Python backend bound to localhost only

## Need Help?

- GitHub Issues: https://github.com/Nythbran23/MSK2K/issues
- Discussions: https://github.com/Nythbran23/MSK2K/discussions

---

Happy coding! 73 de GW4WND

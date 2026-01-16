# MSK2K Icon Setup

The application needs icons for each platform. Place them in the `build/` directory:

## Required Icons

### macOS - icon.icns
- Source: 1024x1024 PNG
- Format: .icns (macOS icon set)
- Contains multiple sizes: 16, 32, 64, 128, 256, 512, 1024

**Create with:**
```bash
# Create iconset folder
mkdir icon.iconset
cp icon-1024.png icon.iconset/icon_512x512@2x.png
# ... add all sizes
iconutil -c icns icon.iconset
mv icon.icns build/
```

Or use online converters like https://cloudconvert.com/png-to-icns

### Windows - icon.ico
- Source: 256x256 PNG
- Format: .ico (Windows icon)
- Contains: 16, 32, 48, 256 sizes

**Create with ImageMagick:**
```bash
convert icon.png -define icon:auto-resize=256,128,64,48,32,16 build/icon.ico
```

Or use online converters

### Linux - icon.png
- Format: PNG
- Size: 512x512 or 1024x1024
- Simply copy your source PNG:
```bash
cp icon.png build/icon.png
```

## Design Guidelines

### MSK2K Icon Suggestions:
- Radio wave symbol
- Meteor trail graphic
- Callsign letters "MSK2K"
- Amateur radio theme
- Clear at small sizes (16x16)

### Colors:
- High contrast for visibility
- Works well on light and dark backgrounds
- Consider The DX Shop branding

## Temporary Placeholder

Until you create custom icons, you can:
1. Use Electron's default icon (no icon files needed)
2. Download a placeholder and rename it
3. Create a simple icon with any graphic editor

The app will build without icons, they're optional but recommended for distribution.

## Tools

- **macOS**: Preview, Pixelmator, Affinity Designer
- **Windows**: Paint.NET, GIMP, Photoshop
- **Linux**: GIMP, Inkscape
- **Online**: Figma, Canva, CloudConvert

## Testing Icons

After adding icons:
```bash
npm run dist
```

Check the built application to verify icons appear correctly in:
- Application bundle/installer
- Task bar / Dock
- Window title bar
- Alt-Tab switcher
- About dialog

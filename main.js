const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');
const fs = require('fs');

let mainWindow = null;
let pythonProcess = null;
const PORT = 8088;
const HOST = '127.0.0.1';

const isDev = !app.isPackaged;

// Get platform key for runtime selection
function platformKey() {
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform === 'win32') return 'win-x64';
  if (platform === 'darwin') {
    return arch === 'arm64' ? 'mac-arm64' : 'mac-x64';
  }
  return 'linux-x64';
}

// Recursively copy directory
function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const dstPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, dstPath);
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
  }
}

// Ensure bundled Python runtime is extracted to userData
function ensureBundledPython() {
  const key = platformKey();
  console.log(`Platform key: ${key}`);
  
  // In development, use system Python
  if (isDev) {
    return null;
  }
  
  const bundledPath = path.join(process.resourcesPath, 'python-runtime', key);
  const targetPath = path.join(app.getPath('userData'), 'python');
  const versionFile = path.join(targetPath, 'runtime_version.txt');
  const bundledVersionFile = path.join(bundledPath, 'runtime_version.txt');
  
  console.log(`Bundled runtime path: ${bundledPath}`);
  console.log(`Target runtime path: ${targetPath}`);
  
  // Check if bundled runtime exists
  if (!fs.existsSync(bundledPath)) {
    console.error(`Bundled Python runtime not found at: ${bundledPath}`);
    dialog.showErrorBox(
      'Installation Error',
      `Python runtime for ${key} not found. Please reinstall MSK2K.`
    );
    app.quit();
    return null;
  }
  
  // Check if extraction is needed
  const bundledVersion = fs.existsSync(bundledVersionFile) 
    ? fs.readFileSync(bundledVersionFile, 'utf8').trim() 
    : 'v1';
  const currentVersion = fs.existsSync(versionFile) 
    ? fs.readFileSync(versionFile, 'utf8').trim() 
    : '';
  
  if (currentVersion !== bundledVersion) {
    console.log(`Extracting Python runtime (${bundledVersion})...`);
    
    // Remove old runtime
    if (fs.existsSync(targetPath)) {
      fs.rmSync(targetPath, { recursive: true, force: true });
    }
    
    // Copy bundled runtime
    copyDir(bundledPath, targetPath);
    console.log('Python runtime extracted successfully');
  } else {
    console.log('Python runtime already up to date');
  }
  
  return targetPath;
}

// Get Python executable path
function getPythonExe(runtimePath) {
  if (isDev) {
    // Development: use system Python
    return process.platform === 'win32' ? 'python' : 'python3';
  }
  
  // Production: use bundled Python
  if (process.platform === 'win32') {
    return path.join(runtimePath, 'Scripts', 'python.exe');
  } else {
    return path.join(runtimePath, 'bin', 'python3');
  }
}

// Get backend script path
function getBackendScript(runtimePath) {
  if (isDev) {
    return path.join(__dirname, 'python', 'msk2k_audio_qso_server_Q12.py');
  }
  return path.join(runtimePath, 'app', 'msk2k_audio_qso_server_Q12.py');
}

// Check if port is available
function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        resolve(false);
      } else {
        resolve(false);
      }
    });
    
    server.once('listening', () => {
      server.close();
      resolve(true);
    });
    
    server.listen(port, HOST);
  });
}

// Start Python backend
async function startPythonBackend() {
  console.log('Starting Python backend...');
  
  // Check if port is available
  const available = await isPortAvailable(PORT);
  if (!available) {
    console.error(`Port ${PORT} is already in use`);
    dialog.showErrorBox(
      'Port Already In Use',
      `MSK2K cannot start because port ${PORT} is already in use.\n\nPlease close any other MSK2K instances or applications using this port.`
    );
    app.quit();
    return false;
  }
  
  // Extract/verify Python runtime
  const runtimePath = ensureBundledPython();
  
  if (!isDev && !runtimePath) {
    return false;
  }
  
  const pythonExe = getPythonExe(runtimePath);
  const backendScript = getBackendScript(runtimePath);
  
  console.log('Python executable:', pythonExe);
  console.log('Backend script:', backendScript);
  
  const args = [
    backendScript,
    '--host', HOST,
    '--port', PORT.toString()
  ];
  
  const workingDir = isDev 
    ? path.join(__dirname, 'python')
    : path.join(runtimePath, 'app');
  
  console.log('Working directory:', workingDir);
  console.log('Starting backend with args:', args);
  
  // Start backend process
  pythonProcess = spawn(pythonExe, args, {
    cwd: workingDir,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      MSK2K_ADIF_PATH: path.join(app.getPath('userData'), 'msk2k_qsos.adi')
    }
  });
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Backend]: ${data}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Backend Error]: ${data}`);
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    if (code !== 0 && code !== null) {
      dialog.showErrorBox(
        'Backend Error',
        `MSK2K backend stopped unexpectedly (exit code: ${code}).\n\nPlease check that all required packages are installed.`
      );
    }
  });
  
  // Wait for backend to be ready
  const ready = await waitForBackend();
  return ready;
}

// Wait for Python backend to be ready
function waitForBackend(maxAttempts = 30) {
  return new Promise((resolve) => {
    let attempts = 0;
    
    const checkBackend = () => {
      const client = net.createConnection({ port: PORT, host: HOST }, () => {
        client.end();
        console.log('Backend is ready!');
        resolve(true);
      });
      
      client.on('error', () => {
        attempts++;
        if (attempts >= maxAttempts) {
          console.error('Backend failed to start');
          dialog.showErrorBox(
            'Backend Failed',
            'MSK2K backend failed to start. Please try restarting the application.'
          );
          resolve(false);
        } else {
          setTimeout(checkBackend, 1000);
        }
      });
    };
    
    setTimeout(checkBackend, 500);
  });
}

// Stop Python backend
function stopPythonBackend() {
  if (pythonProcess) {
    console.log('Stopping Python backend...');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// Create main window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    title: 'MSK2K - Meteor Scatter Digital Mode',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    backgroundColor: '#111111'
  });
  
  mainWindow.loadURL(`http://${HOST}:${PORT}`);
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  
  createMenu();
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'MSK2K',
      submenu: [
        {
          label: 'About MSK2K',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About MSK2K',
              message: 'MSK2K v0.2.0-alpha',
              detail: 'Meteor Scatter Digital Mode for Amateur Radio\n\nMSK2K adaptation by Roger Banks (GW4WND)\nBased on PSK2K by Klaus von der Heide (DJ5HG)\n\nhttps://github.com/Nythbran23/MSK2K'
            });
          }
        },
        { type: 'separator' },
        {
          label: 'Open Log File',
          click: () => {
            const logPath = path.join(app.getPath('userData'), 'msk2k_qsos.adi');
            shell.showItemInFolder(logPath);
          }
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'close' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'GitHub Repository',
          click: () => shell.openExternal('https://github.com/Nythbran23/MSK2K')
        },
        {
          label: 'The DX Shop',
          click: () => shell.openExternal('https://thedxshop.com')
        }
      ]
    }
  ];
  
  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App ready
app.whenReady().then(async () => {
  const started = await startPythonBackend();
  if (started) {
    createWindow();
  }
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopPythonBackend();
    app.quit();
  }
});

// Clean up before quit
app.on('before-quit', () => {
  stopPythonBackend();
});

// Handle errors
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
});
// Force rebuild Fri 16 Jan 2026 14:02:10 GMT

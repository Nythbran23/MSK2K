const { app, BrowserWindow, Menu, Tray, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');

let mainWindow = null;
let pythonProcess = null;
let tray = null;
const PORT = 8089;
const HOST = '127.0.0.1';

// Determine if we're in development or production
const isDev = !app.isPackaged;

// Get the Python backend path
function getPythonBackendPath() {
  if (isDev) {
    return path.join(__dirname, 'python');
  } else {
    // In production, Python files are in resources
    return path.join(process.resourcesPath, 'python');
  }
}

// Find Python executable
function getPythonCommand() {
  const platform = process.platform;
  
  if (platform === 'win32') {
    return 'python';  // Will try python, python3
  } else {
    return 'python3';
  }
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
  const backendPath = getPythonBackendPath();
  const serverScript = path.join(backendPath, 'msk2k_audio_qso_server_Q12.py');
  const pythonCmd = getPythonCommand();
  
  console.log('Starting Python backend...');
  console.log('Backend path:', backendPath);
  console.log('Server script:', serverScript);
  console.log('Python command:', pythonCmd);
  
  // Check if port is available
  const available = await isPortAvailable(PORT);
  if (!available) {
    console.error(`Port ${PORT} is already in use`);
    dialog.showErrorBox(
      'Port Already In Use',
      `MSK2K cannot start because port ${PORT} is already in use.\n\nPlease close any other MSK2K instances or applications using this port.`
    );
    app.quit();
    return;
  }
  
  // Start Python process
  pythonProcess = spawn(pythonCmd, [
    serverScript,
    '--host', HOST,
    '--port', PORT.toString()
  ], {
    cwd: backendPath,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      MSK2K_ADIF_PATH: path.join(app.getPath('userData'), 'msk2k_qsos.adi')
    }
  });
  
  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python]: ${data}`);
  });
  
  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error]: ${data}`);
  });
  
  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    if (code !== 0 && code !== null) {
      dialog.showErrorBox(
        'Backend Error',
        `MSK2K backend stopped unexpectedly (exit code: ${code}).\n\nPlease check that Python 3 and required packages (numpy, scipy, aiohttp, sounddevice) are installed.`
      );
    }
  });
  
  // Wait for backend to be ready
  await waitForBackend();
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
            'MSK2K backend failed to start.\n\nPlease ensure Python 3 and required packages are installed:\n\npip3 install numpy scipy aiohttp sounddevice'
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
  
  // Load the web UI
  mainWindow.loadURL(`http://${HOST}:${PORT}`);
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  
  // Create application menu
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
              message: 'MSK2K v0.1.0',
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
  await startPythonBackend();
  createWindow();
  
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

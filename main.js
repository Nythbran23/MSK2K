const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'build', 'icon.png')
  });

  // Load your HTML file
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'msk2k_audio_qso_ui_Q12.html'));
  
  // Start Python backend
  startPythonBackend();
}

function startPythonBackend() {
  // Determine Python path based on platform and whether we're in dev or production
  const isDev = !app.isPackaged;
  let pythonPath;
  let scriptPath;
  
  if (isDev) {
    // Development mode - use system Python
    pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    scriptPath = path.join(__dirname, 'python', 'msk2k_audio_qso_server_Q12.py');
  } else {
    // Production mode - use bundled Python
    if (process.platform === 'win32') {
      pythonPath = path.join(process.resourcesPath, 'python', 'python.exe');
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    } else if (process.platform === 'darwin') {
      pythonPath = path.join(
        process.resourcesPath, 
        'python', 
        'Python.framework', 
        'Versions', 
        '3.11', 
        'bin', 
        'python3'
      );
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    } else {
      // Linux
      pythonPath = path.join(process.resourcesPath, 'python', 'bin', 'python3');
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    }
  }

  console.log('Starting Python backend...');
  console.log('Mode:', isDev ? 'Development' : 'Production');
  console.log('Platform:', process.platform);
  console.log('Python path:', pythonPath);
  console.log('Script path:', scriptPath);
  console.log('Resources path:', process.resourcesPath);

  pythonProcess = spawn(pythonPath, [scriptPath], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python] ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Error] ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    if (code !== 0 && code !== null) {
      console.error('Python backend crashed!');
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python backend:', err);
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Cleanup on app quit
app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

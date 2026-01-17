const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'msk2k_audio_qso_ui_Q12.html'));
  
  // Open DevTools
  mainWindow.webContents.openDevTools();
  
  // Wait a bit for window to be ready, then start Python
  setTimeout(() => {
    startPythonBackend();
  }, 1000);
}

function logToConsole(msg) {
  console.log(msg);
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.executeJavaScript(`console.log(${JSON.stringify(msg)})`);
  }
}

function startPythonBackend() {
  const isDev = !app.isPackaged;
  let pythonPath;
  let scriptPath;
  
  logToConsole('=== STARTING PYTHON BACKEND ===');
  logToConsole(`Mode: ${isDev ? 'Development' : 'Production'}`);
  logToConsole(`Platform: ${process.platform}`);
  logToConsole(`Packaged: ${app.isPackaged}`);
  logToConsole(`__dirname: ${__dirname}`);
  logToConsole(`process.resourcesPath: ${process.resourcesPath}`);
  
  if (isDev) {
    pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    scriptPath = path.join(__dirname, 'python', 'msk2k_server_wrapper.py');
  } else {
    if (process.platform === 'win32') {
      pythonPath = path.join(process.resourcesPath, 'python', 'python.exe');
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_server_wrapper.py');
    } else if (process.platform === 'darwin') {
      pythonPath = 'python3';
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_server_wrapper.py');
    } else {
      pythonPath = 'python3';
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_server_wrapper.py');
    }
  }

  logToConsole(`Python path: ${pythonPath}`);
  logToConsole(`Script path: ${scriptPath}`);
  
  // Check if files exist
  if (process.platform === 'win32' && !isDev) {
    const pythonExists = fs.existsSync(pythonPath);
    const scriptExists = fs.existsSync(scriptPath);
    logToConsole(`Python exists: ${pythonExists}`);
    logToConsole(`Script exists: ${scriptExists}`);
    
    if (!pythonExists) {
      dialog.showErrorBox('Python Not Found', `Python executable not found at: ${pythonPath}`);
      return;
    }
    if (!scriptExists) {
      dialog.showErrorBox('Script Not Found', `Python script not found at: ${scriptPath}`);
      return;
    }
  }

  logToConsole('Spawning Python process...');
  
  try {
    pythonProcess = spawn(pythonPath, [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env }
    });

    logToConsole('Python process spawned successfully');

    pythonProcess.stdout.on('data', (data) => {
      const msg = `[Python STDOUT] ${data.toString().trim()}`;
      logToConsole(msg);
    });

    pythonProcess.stderr.on('data', (data) => {
      const msg = `[Python STDERR] ${data.toString().trim()}`;
      logToConsole(msg);
    });

    pythonProcess.on('close', (code) => {
      logToConsole(`Python process exited with code ${code}`);
    });

    pythonProcess.on('error', (err) => {
      logToConsole(`Failed to start Python: ${err.message}`);
      dialog.showErrorBox('Python Error', `Failed to start Python backend: ${err.message}\n\nPath: ${pythonPath}`);
    });
  } catch (err) {
    logToConsole(`Exception spawning Python: ${err.message}`);
    dialog.showErrorBox('Exception', `Exception starting Python: ${err.message}`);
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

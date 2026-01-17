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
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'msk2k_audio_qso_ui_Q12.html'));
  
  // Open DevTools to see console logs
  mainWindow.webContents.openDevTools();
  
  startPythonBackend();
}

function startPythonBackend() {
  const isDev = !app.isPackaged;
  let pythonPath;
  let scriptPath;
  
  if (isDev) {
    pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    scriptPath = path.join(__dirname, 'python', 'msk2k_audio_qso_server_Q12.py');
  } else {
    if (process.platform === 'win32') {
      pythonPath = path.join(process.resourcesPath, 'python', 'python.exe');
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    } else if (process.platform === 'darwin') {
      pythonPath = 'python3';  // Mac uses system Python
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    } else {
      pythonPath = 'python3';  // Linux uses system Python
      scriptPath = path.join(process.resourcesPath, 'python-app', 'msk2k_audio_qso_server_Q12.py');
    }
  }

  console.log('=== Starting Python Backend ===');
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
    console.log('[Python]', data.toString().trim());
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error('[Python Error]', data.toString().trim());
  });

  pythonProcess.on('close', (code) => {
    console.log('Python process exited with code', code);
  });

  pythonProcess.on('error', (err) => {
    console.error('Failed to start Python:', err);
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

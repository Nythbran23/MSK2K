const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');

module.exports = async function(context) {
  const platform = context.electronPlatformName;
  const appOutDir = context.appOutDir;
  
  console.log(`\n=== Bundling Python for ${platform} ===\n`);
  
  if (platform === 'win32') {
    await bundleWindowsPython(appOutDir);
  } else if (platform === 'darwin') {
    await bundleMacPython(appOutDir);
  } else if (platform === 'linux') {
    await bundleLinuxPython(appOutDir);
  }
  
  console.log(`\n=== ${platform} Python bundling complete ===\n`);
};

async function bundleWindowsPython(appOutDir) {
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-embed-amd64.zip`;
  const zipPath = path.join(resourcesDir, 'python-embed.zip');
  
  console.log('Downloading Python for Windows...');
  execSync(`curl -L -o "${zipPath}" "${pythonUrl}"`, { stdio: 'inherit' });
  
  execSync(`powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${pythonDir}' -Force"`, { stdio: 'inherit' });
  await fs.remove(zipPath);

  const pthFile = path.join(pythonDir, 'python311._pth');
  let pthContent = await fs.readFile(pthFile, 'utf-8');
  pthContent = pthContent.replace('#import site', 'import site');
  await fs.writeFile(pthFile, pthContent);

  console.log('Installing pip...');
  const getPipPath = path.join(pythonDir, 'get-pip.py');
  execSync(`curl -L -o "${getPipPath}" https://bootstrap.pypa.io/get-pip.py`, { stdio: 'inherit' });
  execSync(`"${path.join(pythonDir, 'python.exe')}" "${getPipPath}"`, { stdio: 'inherit', cwd: pythonDir });

  console.log('Installing dependencies...');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  execSync(`"${path.join(pythonDir, 'python.exe')}" -m pip install -r "${requirementsPath}"`, { stdio: 'inherit', cwd: pythonDir });

  console.log('Copying Python app...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
  );
}

async function bundleMacPython(appOutDir) {
  // For macOS, appOutDir might be the .app itself or dist/mac-arm64
  // We need to find the actual .app/Contents/Resources directory
  let resourcesDir;
  
  console.log(`Mac appOutDir: ${appOutDir}`);
  
  // Check if appOutDir ends with .app
  if (appOutDir.endsWith('.app')) {
    resourcesDir = path.join(appOutDir, 'Contents', 'Resources');
  } else {
    // appOutDir is dist/mac-arm64, find the .app
    const files = await fs.readdir(appOutDir);
    const appFile = files.find(f => f.endsWith('.app'));
    if (!appFile) {
      throw new Error('Could not find .app bundle in ' + appOutDir);
    }
    resourcesDir = path.join(appOutDir, appFile, 'Contents', 'Resources');
  }
  
  console.log(`Mac resourcesDir: ${resourcesDir}`);
  
  const pythonDir = path.join(resourcesDir, 'python');
  await fs.ensureDir(pythonDir);

  // Download standalone Python
  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-macos11.pkg`;
  const pkgPath = path.join(resourcesDir, 'python-installer.pkg');
  
  console.log(`Downloading Python ${pythonVersion} for macOS...`);
  execSync(`curl -L -o "${pkgPath}" "${pythonUrl}"`, { stdio: 'inherit' });
  
  console.log('Extracting Python.framework...');
  const extractDir = path.join(resourcesDir, 'python-extract');
  
  // Clean up extract dir if it exists (use sync to ensure it's done before pkgutil)
  if (fs.pathExistsSync(extractDir)) {
    fs.removeSync(extractDir);
  }
  
  await fs.ensureDir(extractDir);
  
  // Extract the pkg
  execSync(`pkgutil --expand "${pkgPath}" "${extractDir}"`, { stdio: 'inherit' });
  
  // Extract the Python framework payload
  const payloadPath = path.join(extractDir, 'Python_Framework.pkg', 'Payload');
  execSync(`tar -xzf "${payloadPath}" -C "${pythonDir}"`, { stdio: 'inherit' });
  
  // Clean up
  await fs.remove(extractDir);
  await fs.remove(pkgPath);

  const pythonBin = path.join(pythonDir, 'Library', 'Frameworks', 'Python.framework', 'Versions', '3.11', 'bin', 'python3');
  const pipBin = path.join(pythonDir, 'Library', 'Frameworks', 'Python.framework', 'Versions', '3.11', 'bin', 'pip3');
  
  console.log('Upgrading pip...');
  execSync(`"${pythonBin}" -m pip install --upgrade pip`, { stdio: 'inherit' });
  
  console.log('Installing dependencies...');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  execSync(`"${pipBin}" install -r "${requirementsPath}"`, { stdio: 'inherit' });
  
  console.log('Copying Python app...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
  );
  
  console.log('macOS Python bundle created successfully');
}

async function bundleLinuxPython(appOutDir) {
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  console.log('Creating Python virtual environment for Linux...');
  
  // Create venv with --copies to avoid symlinks to system Python
  execSync('python3 -m venv --copies ' + pythonDir, { stdio: 'inherit' });
  
  const pythonBin = path.join(pythonDir, 'bin', 'python3');
  const pipBin = path.join(pythonDir, 'bin', 'pip3');
  
  console.log('Upgrading pip...');
  execSync(`"${pythonBin}" -m pip install --upgrade pip`, { stdio: 'inherit' });
  
  console.log('Installing dependencies...');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  execSync(`"${pipBin}" install -r "${requirementsPath}"`, { stdio: 'inherit' });
  
  console.log('Copying Python app...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
  );
  
  console.log('Linux Python bundle created successfully');
}

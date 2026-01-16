const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Electron Builder afterPack hook
 * Bundles Python runtime into the packaged app
 */
module.exports = async function(context) {
  const appOutDir = context.appOutDir;
  const platform = context.electronPlatformName;
  const arch = context.arch;
  
  console.log(`\n=== Bundling Python for ${platform}-${arch} ===\n`);

  if (platform === 'win32') {
    await bundleWindowsPython(appOutDir);
  } else if (platform === 'darwin') {
    await bundleMacPython(appOutDir);
  } else if (platform === 'linux') {
    await bundleLinuxPython(appOutDir);
  }

  console.log(`\n=== Python bundling complete for ${platform}-${arch} ===\n`);
};

async function bundleWindowsPython(appOutDir) {
  console.log('Bundling Python for Windows...');
  
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-embed-amd64.zip`;
  const zipPath = path.join(resourcesDir, 'python-embed.zip');
  
  console.log(`Downloading Python ${pythonVersion} embeddable...`);
  execSync(`curl -L -o "${zipPath}" "${pythonUrl}"`, { stdio: 'inherit' });
  
  console.log('Extracting Python...');
  execSync(`powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${pythonDir}' -Force"`, 
    { stdio: 'inherit' });
  
  await fs.remove(zipPath);

  const pthFile = path.join(pythonDir, 'python311._pth');
  let pthContent = await fs.readFile(pthFile, 'utf-8');
  pthContent = pthContent.replace('#import site', 'import site');
  await fs.writeFile(pthFile, pthContent);

  console.log('Installing pip...');
  const getPipPath = path.join(pythonDir, 'get-pip.py');
  execSync(`curl -L -o "${getPipPath}" https://bootstrap.pypa.io/get-pip.py`, 
    { stdio: 'inherit' });
  
  execSync(`"${path.join(pythonDir, 'python.exe')}" "${getPipPath}"`, 
    { stdio: 'inherit', cwd: pythonDir });

  console.log('Installing Python dependencies...');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  
  execSync(
    `"${path.join(pythonDir, 'python.exe')}" -m pip install -r "${requirementsPath}"`,
    { stdio: 'inherit', cwd: pythonDir }
  );

  console.log('Copying Python application files...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { 
      filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc')
    }
  );

  console.log('Windows Python bundling complete!');
}

async function bundleMacPython(appOutDir) {
  console.log('Bundling Python for macOS...');
  
  const resourcesDir = path.join(appOutDir, '..', 'Resources');
  const pythonDir = path.join(resourcesDir, 'python');
  const extractDir = path.join(resourcesDir, 'python-extract');
  
  // CRITICAL: Use shell rm -rf to force synchronous removal
  console.log('Cleaning any existing directories...');
  execSync(`rm -rf "${pythonDir}" "${extractDir}"`, { stdio: 'inherit' });
  
  await fs.ensureDir(pythonDir);

  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-macos11.pkg`;
  const pkgPath = path.join(resourcesDir, 'python-installer.pkg');
  
  console.log(`Downloading Python ${pythonVersion}...`);
  execSync(`curl -L -o "${pkgPath}" "${pythonUrl}"`, { stdio: 'inherit' });
  
  console.log('Extracting Python framework...');
  await fs.ensureDir(extractDir);
  
  execSync(`pkgutil --expand "${pkgPath}" "${extractDir}"`, { stdio: 'inherit' });
  execSync(`tar -xzf "${path.join(extractDir, 'Python_Framework.pkg', 'Payload')}" -C "${pythonDir}"`, 
    { stdio: 'inherit' });
  
  execSync(`rm -rf "${extractDir}" "${pkgPath}"`, { stdio: 'inherit' });

  console.log('Installing Python dependencies...');
  const pythonBin = path.join(pythonDir, 'Python.framework', 'Versions', '3.11', 'bin', 'python3');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  
  execSync(`"${pythonBin}" -m pip install --upgrade pip`, { stdio: 'inherit' });
  execSync(`"${pythonBin}" -m pip install -r "${requirementsPath}"`, { stdio: 'inherit' });

  console.log('Copying Python application files...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { 
      filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc')
    }
  );

  console.log('macOS Python bundling complete!');
}

async function bundleLinuxPython(appOutDir) {
  console.log('Bundling Python for Linux...');
  
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  console.log('Creating Python virtual environment...');
  execSync(`python3 -m venv "${pythonDir}"`, { stdio: 'inherit' });

  console.log('Installing Python dependencies...');
  const pipPath = path.join(pythonDir, 'bin', 'pip');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  
  execSync(`"${pipPath}" install --upgrade pip`, { stdio: 'inherit' });
  execSync(`"${pipPath}" install -r "${requirementsPath}"`, { stdio: 'inherit' });

  console.log('Copying Python application files...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { 
      filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc')
    }
  );

  console.log('Linux Python bundling complete!');
}

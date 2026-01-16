const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');

module.exports = async function(context) {
  const platform = context.electronPlatformName;
  
  console.log(`\n=== Bundling Python for ${platform} ===\n`);

  if (platform === 'win32') {
    await bundleWindowsPython(context.appOutDir);
  } else if (platform === 'darwin') {
    await bundleMacPython(context.appOutDir);
  } else if (platform === 'linux') {
    await bundleLinuxPython(context.appOutDir);
  }

  console.log(`\n=== Complete ===\n`);
};

async function bundleWindowsPython(appOutDir) {
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-embed-amd64.zip`;
  const zipPath = path.join(resourcesDir, 'python-embed.zip');
  
  console.log('Downloading Python embeddable...');
  execSync(`curl -L -o "${zipPath}" "${pythonUrl}"`, { stdio: 'inherit' });
  
  console.log('Extracting...');
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

  console.log('Copying Python app files...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
  );

  console.log('Windows Python complete!');
}

async function bundleMacPython(appOutDir) {
  console.log('Bundling Python for macOS...');
  
  const resourcesDir = path.join(appOutDir, '..', 'Resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  // Use a truly unique temp directory in /tmp
  const uniqueId = `${Date.now()}-${process.pid}`;
  const tmpDir = `/tmp/msk2k-python-${uniqueId}`;
  
  console.log(`Using temp directory: ${tmpDir}`);
  execSync(`mkdir -p "${tmpDir}"`, { stdio: 'inherit' });
  
  try {
    const pythonVersion = '3.11.9';
    const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-macos11.pkg`;
    const pkgPath = path.join(tmpDir, 'python.pkg');
    
    console.log('Downloading Python...');
    execSync(`curl -L -o "${pkgPath}" "${pythonUrl}"`, { stdio: 'inherit' });
    
    console.log('Extracting Python framework...');
    // Extract directly without pkgutil intermediate step
    execSync(`cd "${tmpDir}" && xar -xf "${pkgPath}"`, { stdio: 'inherit' });
    
    // Clean and create python directory
    execSync(`rm -rf "${pythonDir}"`, { stdio: 'inherit' });
    execSync(`mkdir -p "${pythonDir}"`, { stdio: 'inherit' });
    
    // Extract the payload tar
    execSync(`tar -xzf "${tmpDir}/Python_Framework.pkg/Payload" -C "${pythonDir}"`, { stdio: 'inherit' });
    
    console.log('Installing dependencies...');
    const pythonBin = path.join(pythonDir, 'Python.framework', 'Versions', '3.11', 'bin', 'python3');
    const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
    
    execSync(`"${pythonBin}" -m pip install --upgrade pip`, { stdio: 'inherit' });
    execSync(`"${pythonBin}" -m pip install -r "${requirementsPath}"`, { stdio: 'inherit' });

    console.log('Copying Python app files...');
    await fs.copy(
      path.join(__dirname, '..', 'python'),
      path.join(resourcesDir, 'python-app'),
      { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
    );
    
  } finally {
    // Clean up temp directory
    execSync(`rm -rf "${tmpDir}"`, { stdio: 'inherit' });
  }

  console.log('macOS Python complete!');
}

async function bundleLinuxPython(appOutDir) {
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  console.log('Creating Python venv...');
  execSync(`python3 -m venv "${pythonDir}"`, { stdio: 'inherit' });

  console.log('Installing dependencies...');
  const pipPath = path.join(pythonDir, 'bin', 'pip');
  const requirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
  
  execSync(`"${pipPath}" install --upgrade pip`, { stdio: 'inherit' });
  execSync(`"${pipPath}" install -r "${requirementsPath}"`, { stdio: 'inherit' });

  console.log('Copying Python app files...');
  await fs.copy(
    path.join(__dirname, '..', 'python'),
    path.join(resourcesDir, 'python-app'),
    { filter: (src) => !src.includes('__pycache__') && !src.includes('.pyc') }
  );

  console.log('Linux Python complete!');
}

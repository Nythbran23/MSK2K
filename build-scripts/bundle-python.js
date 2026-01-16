const fs = require('fs-extra');
const path = require('path');
const { execSync } = require('child_process');

module.exports = async function(context) {
  const platform = context.electronPlatformName;
  
  if (platform === 'win32') {
    console.log('\n=== Bundling Python for Windows ===\n');
    await bundleWindowsPython(context.appOutDir);
    console.log('\n=== Windows Python complete ===\n');
  } else {
    console.log(`\n${platform} - Skipping Python bundling (uses system Python)\n`);
  }
};

async function bundleWindowsPython(appOutDir) {
  const resourcesDir = path.join(appOutDir, 'resources');
  const pythonDir = path.join(resourcesDir, 'python');
  
  await fs.ensureDir(pythonDir);

  const pythonVersion = '3.11.9';
  const pythonUrl = `https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-embed-amd64.zip`;
  const zipPath = path.join(resourcesDir, 'python-embed.zip');
  
  console.log('Downloading Python...');
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

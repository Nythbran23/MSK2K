# Git Workflow for MSK2K

## Initial Push to GitHub

Your repo is at: https://github.com/Nythbran23/MSK2K

### First Time Setup

```bash
# Navigate to the project directory
cd /path/to/msk2k-electron

# Initialize git (if not already done)
git init

# Add the GitHub remote
git remote add origin https://github.com/Nythbran23/MSK2K.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: Electron-based MSK2K application"

# Push to GitHub
git push -u origin main
```

If the repo already has content, you might need:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Daily Workflow

### Making Changes

```bash
# Check what changed
git status

# Stage specific files
git add main.js python/msk2k_complete.py

# Or stage everything
git add .

# Commit with message
git commit -m "Add feature: improved audio level monitoring"

# Push to GitHub
git push
```

### Commit Message Style

Good commit messages:
- `feat: Add auto-RX gain control`
- `fix: Resolve audio device selection on Windows`
- `docs: Update README with Linux instructions`
- `refactor: Simplify Python backend startup`
- `perf: Optimize MSK decoder performance`

## Branches

### Creating a Feature Branch

```bash
# Create and switch to new branch
git checkout -b feature/new-audio-system

# Make changes, commit
git add .
git commit -m "feat: new audio routing system"

# Push branch
git push -u origin feature/new-audio-system
```

Then create a Pull Request on GitHub.

### Merging Back

```bash
# Switch to main
git checkout main

# Pull latest
git pull

# Merge feature
git merge feature/new-audio-system

# Push
git push
```

## Releases

### Creating a Release

```bash
# Make sure main is clean and pushed
git status
git push

# Create and push tag
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

**GitHub Actions will automatically:**
1. Build for macOS, Windows, Linux
2. Create a GitHub Release
3. Attach all installers to the release

### Viewing Releases

Visit: https://github.com/Nythbran23/MSK2K/releases

## Common Commands

```bash
# See commit history
git log --oneline

# See what changed in a file
git diff python/msk2k_complete.py

# Undo changes to a file (before commit)
git checkout -- main.js

# Undo last commit (keep changes)
git reset --soft HEAD~1

# See all branches
git branch -a

# Delete local branch
git branch -d feature/old-feature

# Update from GitHub
git pull
```

## .gitignore

Already configured to ignore:
- `node_modules/` - NPM packages (huge, not needed in repo)
- `dist/` - Build outputs (generated, not source)
- `build/` - Temporary build files
- `__pycache__/` - Python bytecode
- `.DS_Store` - macOS metadata
- `*.log` - Log files

## Collaborating

### Cloning the Repo (Fresh Start)

```bash
git clone https://github.com/Nythbran23/MSK2K.git
cd MSK2K
npm install
pip3 install -r python/requirements.txt
npm start
```

### Pull Request Workflow

1. Fork the repo (or branch)
2. Make changes
3. Push to your fork/branch
4. Open Pull Request on GitHub
5. Review and merge

## GitHub Actions

Status: https://github.com/Nythbran23/MSK2K/actions

The CI/CD pipeline runs on:
- Every push to `main`
- Every pull request
- Every tag push (v*)
- Manual trigger

Check the Actions tab on GitHub to see build status.

## Troubleshooting

### Authentication Issues

Use personal access token:
```bash
# GitHub removed password auth
# Create token at: https://github.com/settings/tokens
# Use token as password when prompted
```

Or setup SSH:
```bash
# Generate key
ssh-keygen -t ed25519 -C "your@email.com"

# Add to GitHub: https://github.com/settings/keys

# Change remote to SSH
git remote set-url origin git@github.com:Nythbran23/MSK2K.git
```

### Large Files

If you accidentally commit large files:
```bash
# Remove from staging
git rm --cached large-file.bin

# Add to .gitignore
echo "large-file.bin" >> .gitignore

# Commit
git commit -m "Remove large file"
```

### Merge Conflicts

```bash
# Pull latest
git pull

# If conflicts, edit files to resolve
# Look for <<<<<<< and >>>>>>>

# After resolving
git add .
git commit -m "Resolve merge conflict"
git push
```

## Quick Reference

```bash
git status              # See what changed
git add .               # Stage all changes
git commit -m "msg"     # Commit with message
git push                # Push to GitHub
git pull                # Get latest from GitHub
git log                 # See history
git branch              # List branches
git checkout -b name    # Create branch
git tag -a v1.0.0       # Create tag
```

## Help

- Git Book: https://git-scm.com/book
- GitHub Docs: https://docs.github.com
- Git Cheat Sheet: https://training.github.com/downloads/github-git-cheat-sheet/

---

Need help? Open an issue: https://github.com/Nythbran23/MSK2K/issues

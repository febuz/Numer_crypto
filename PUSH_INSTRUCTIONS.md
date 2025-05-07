# Repository Push Instructions

This document provides instructions for pushing the updated Numer_crypto repository to GitHub, which will replace the existing files with our new structure.

## Using GitHub CLI (Recommended)

If you have the GitHub CLI (`gh`) installed, it's the easiest way to push:

1. Login to GitHub:
   ```
   gh auth login
   ```

2. Clone the repository (if not already done):
   ```
   gh repo clone febuz/Numer_crypto
   cd Numer_crypto
   ```

3. Check out a new branch (recommended to avoid directly pushing to main):
   ```
   git checkout -b updated_codebase
   ```

4. Add all files in the current directory:
   ```
   git add .
   ```

5. Commit changes:
   ```
   git commit -m "Complete repository restructure for Numerai Crypto pipeline"
   ```

6. Push changes:
   ```
   git push -u origin updated_codebase
   ```

7. Create a pull request:
   ```
   gh pr create --title "Complete repository restructure" --body "This PR restructures the entire repository with improved code organization, documentation, and implementation of multiple prediction strategies."
   ```

## Using HTTPS (Alternative Method)

If you don't have the GitHub CLI, you can use the standard Git commands with HTTPS:

1. Configure Git credentials:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your-email@example.com"
   git config --global credential.helper store
   ```

2. Clone the repository (if not already done):
   ```
   git clone https://github.com/febuz/Numer_crypto.git
   cd Numer_crypto
   ```

3. Check out a new branch:
   ```
   git checkout -b updated_codebase
   ```

4. Add all files:
   ```
   git add .
   ```

5. Commit changes:
   ```
   git commit -m "Complete repository restructure for Numerai Crypto pipeline"
   ```

6. Push changes:
   ```
   git push -u origin updated_codebase
   ```

7. Go to the GitHub website to create a pull request:
   https://github.com/febuz/Numer_crypto/pull/new/updated_codebase

## Force Push to Replace All Content (Use with Caution)

If you want to completely replace the existing repository content:

1. Clone the repository:
   ```
   git clone https://github.com/febuz/Numer_crypto.git
   cd Numer_crypto
   ```

2. Remove all files and Git history (CAUTION: this is destructive):
   ```
   rm -rf .git
   git init
   ```

3. Configure Git:
   ```
   git config user.name "Your Name"
   git config user.email "your-email@example.com"
   ```

4. Add all files:
   ```
   git add .
   ```

5. Commit:
   ```
   git commit -m "Complete repository restructure for Numerai Crypto pipeline"
   ```

6. Add remote:
   ```
   git remote add origin https://github.com/febuz/Numer_crypto.git
   ```

7. Force push (WARNING: This completely overwrites the remote repository):
   ```
   git push -f origin master
   ```

## Important Notes

1. The force push option will remove any files in the GitHub repository that aren't in your local repository. Use with caution.

2. You may need to create a GitHub Personal Access Token to authenticate if you're using HTTPS:
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Create a token with 'repo' scope
   - Use this token as your password when Git asks for credentials

3. After pushing, verify all files look correct in the GitHub web interface.

4. Consider making a backup of the original repository before force pushing.

## Questions or Issues

If you encounter any issues during the push process, please refer to GitHub's documentation or contact your repository administrator for assistance.
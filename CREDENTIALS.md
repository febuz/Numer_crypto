# Using Environment Variables for Google OAuth Credentials

This document explains how to securely manage Google OAuth credentials for the Numerai Crypto project.

## Setup Instructions

1. Create a `.env` file in the root directory of the project using the provided `.env.example` as a template:
   ```
   cp .env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values with your actual Google OAuth credentials:
   ```
   GOOGLE_CLIENT_ID=your_actual_client_id_here
   GOOGLE_CLIENT_SECRET=your_actual_client_secret_here
   ```

3. Run the `generate_secrets.py` script to create the `client_secrets.json` file from your environment variables:
   ```
   python generate_secrets.py
   ```

4. Verify that `client_secrets.json` has been created correctly.

## Security Notes

- The `.env` file containing your actual credentials is excluded from Git via `.gitignore`
- Never commit the `.env` file or `client_secrets.json` with real credentials to the repository
- The `client_secrets.template.json` and `.env.example` files show the expected format without revealing sensitive information
- If you need to share this project, ensure the recipient creates their own `.env` file with their credentials

## Troubleshooting

If you encounter issues:

1. Ensure the `.env` file exists and contains valid credentials
2. Check that you've run `generate_secrets.py` to create the `client_secrets.json` file
3. Verify that Python's `dotenv` package is installed (`pip install python-dotenv`)

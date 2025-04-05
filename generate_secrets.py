#!/usr/bin/env python3
"""
Script to load environment variables and generate client_secrets.json
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
client_id = os.getenv('GOOGLE_CLIENT_ID')
client_secret = os.getenv('GOOGLE_CLIENT_SECRET')

# Check if credentials are available
if not client_id or not client_secret:
    print("Error: Google OAuth credentials not found in environment variables.")
    print("Please create a .env file with GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
    print("You can use .env.example as a template.")
    exit(1)

# Create client_secrets.json content
client_secrets = {
    "installed": {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uris": ["http://localhost", "urn:ietf:wg:oauth:2.0:oob"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token"
    }
}

# Write to client_secrets.json
with open('client_secrets.json', 'w') as f:
    json.dump(client_secrets, f, indent=2)

print("Successfully generated client_secrets.json from environment variables.")

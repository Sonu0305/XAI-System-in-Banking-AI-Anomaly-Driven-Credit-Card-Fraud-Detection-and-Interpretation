"""
Credentials loader for Kaggle API authentication.
Supports:
1. .env containing path (e.g., ~/.kaggle/access_token)
2. .env containing KAGGLE_TOKEN_PATH / KAGGLE_API_TOKEN
3. .env containing KAGGLE_USERNAME / KAGGLE_KEY
4. Default ~/.kaggle/access_token or ~/.kaggle/kaggle.json
"""

import os
from pathlib import Path
from typing import Optional


def load_kaggle_credentials(env_path: str = ".env") -> bool:
    """
    Locates and configures Kaggle credentials for the Kaggle API.
    Returns True if valid credentials were found and configured.
    """
    token: Optional[str] = None
    token_source: Optional[str] = None

    # 1. Check .env file
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Case A: Plain path line in .env (e.g., ~/.kaggle/access_token)
                if line.startswith("~") or line.startswith("/") or "access_token" in line:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        cand = v.strip().strip("'\"")
                    else:
                        cand = line
                    expanded = os.path.expanduser(cand)
                    if os.path.isfile(expanded):
                        with open(expanded, "r", encoding="utf-8") as tf:
                            token = tf.read().strip()
                        token_source = expanded
                        break

                # Case B: Standard env vars
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    os.environ[k] = v

    # 2. Check environment variables
    if not token and "KAGGLE_API_TOKEN" in os.environ:
        token = os.environ["KAGGLE_API_TOKEN"].strip()
        token_source = "KAGGLE_API_TOKEN environment variable"

    if not token and "KAGGLE_TOKEN_PATH" in os.environ:
        expanded = os.path.expanduser(os.environ["KAGGLE_TOKEN_PATH"])
        if os.path.isfile(expanded):
            with open(expanded, "r", encoding="utf-8") as tf:
                token = tf.read().strip()
            token_source = expanded

    # 3. Check default locations
    if not token:
        default_token_path = os.path.expanduser("~/.kaggle/access_token")
        if os.path.isfile(default_token_path):
            with open(default_token_path, "r", encoding="utf-8") as tf:
                token = tf.read().strip()
            token_source = default_token_path

    # If an access token was located, ensure ~/.kaggle/access_token exists
    if token:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        token_file = kaggle_dir / "access_token"
        if not token_file.exists() or token_file.read_text().strip() != token:
            token_file.write_text(token.strip())
            os.chmod(token_file, 0o600)
        os.environ["KAGGLE_API_TOKEN"] = token
        return True

    # 4. Check for legacy username + key
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return True

    default_json = Path.home() / ".kaggle" / "kaggle.json"
    if default_json.is_file():
        return True

    return False


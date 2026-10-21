"""
deploy_to_github.py
===================
Automated Drive-to-GitHub Repository Creator & Pusher.
Creates a new GitHub repository via the GitHub REST API, uploads all AIOps
project files (including `.devcontainer/devcontainer.json` for GitHub Codespaces),
and prints the direct one-click GitHub Codespaces launch URL.

Usage (from Google Colab, local terminal, or Drive):
    export GITHUB_TOKEN="ghp_your_personal_access_token"
    python deploy_to_github.py --repo-name devops-aiops-remediation --private
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from typing import Dict, List, Optional

import requests

# Core repository files to upload from Google Drive or current directory
REPO_FILES_MAP: Dict[str, str] = {
    "requirements.txt": "requirements.txt",
    "data_loader.py": "data_loader.py",
    "train.py": "train.py",
    "connector.py": "connector.py",
    "remediation.py": "remediation.py",
    "app.py": "app.py",
    "README.md": "README.md",
    "devcontainer.json": ".devcontainer/devcontainer.json",
}

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/devops_aiops"


def resolve_local_or_drive_path(filename: str) -> str:
    """Locate the file in current working directory or mounted Google Drive."""
    candidates = [
        os.path.join(".", filename),
        os.path.join(DRIVE_PROJECT_DIR, filename),
        os.path.join("/content/drive/MyDrive", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in current directory or {DRIVE_PROJECT_DIR}.")


def create_and_push_github_repo(
    github_token: str,
    repo_name: str = "devops-aiops-remediation",
    private: bool = False,
    description: str = "End-to-End AIOps CI/CD & Cloud Failure Detection, Predictive Maintenance, SHAP RCA & Automated Remediation",
) -> Dict[str, str]:
    """Create a GitHub repository and push all project & Codespaces configuration files via REST API."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Fetch authenticated GitHub username
    user_resp = requests.get("https://api.github.com/user", headers=headers, timeout=15)
    if user_resp.status_code != 200:
        raise RuntimeError(f"GitHub authentication failed (HTTP {user_resp.status_code}): {user_resp.text}")
    owner = user_resp.json()["login"]

    # 2. Create repository (or reuse if it already exists)
    create_resp = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json={
            "name": repo_name,
            "description": description,
            "private": private,
            "auto_init": True,
        },
        timeout=15,
    )
    if create_resp.status_code not in (201, 422):
        raise RuntimeError(f"Failed to create repository (HTTP {create_resp.status_code}): {create_resp.text}")

    full_repo = f"{owner}/{repo_name}"
    print(f"[1/3] Repository ready: https://github.com/{full_repo}")

    # 3. Upload each file via GitHub Contents API (`PUT /repos/{owner}/{repo}/contents/{path}`)
    uploaded_files: List[str] = []
    for src_name, dest_repo_path in REPO_FILES_MAP.items():
        try:
            local_path = resolve_local_or_drive_path(src_name)
        except FileNotFoundError:
            continue

        with open(local_path, "rb") as fh:
            encoded_content = base64.b64encode(fh.read()).decode("utf-8")

        contents_url = f"https://api.github.com/repos/{full_repo}/contents/{dest_repo_path}"
        existing_Resp = requests.get(contents_url, headers=headers, timeout=10)
        payload: Dict[str, Optional[str]] = {
            "message": f"chore(aiops): add {dest_repo_path}",
            "content": encoded_content,
            "branch": "main",
        }
        if existing_Resp.status_code == 200:
            payload["sha"] = existing_Resp.json().get("sha")

        put_resp = requests.put(contents_url, headers=headers, json=payload, timeout=20)
        if put_resp.status_code in (200, 201):
            uploaded_files.append(dest_repo_path)
            print(f"  ✓ Uploaded {dest_repo_path}")
        else:
            print(f"  ⚠ Failed to upload {dest_repo_path}: HTTP {put_resp.status_code} {put_resp.text}")

    repo_url = f"https://github.com/{full_repo}"
    codespace_url = f"https://codespaces.new/{full_repo}?quickstart=1"
    print(f"[2/3] Uploaded {len(uploaded_files)} files to {repo_url}")
    print(f"[3/3] Launch directly in GitHub Codespaces: {codespace_url}")

    return {
        "owner": owner,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "codespace_url": codespace_url,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move AIOps files from Google Drive/local to a new GitHub repo.")
    parser.add_argument("--repo-name", default="devops-aiops-remediation", help="Target GitHub repository name")
    parser.add_argument("--private", action="store_true", help="Create as a private GitHub repository")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="GitHub Personal Access Token")
    args = parser.parse_args()

    if not args.token:
        print("Error: Please provide a GitHub Personal Access Token via --token or export GITHUB_TOKEN='ghp_...'")
        sys.exit(1)

    result = create_and_push_github_repo(
        github_token=args.token,
        repo_name=args.repo_name,
        private=args.private,
    )
    print(json.dumps(result, indent=2))

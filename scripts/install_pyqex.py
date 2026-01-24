#!/usr/bin/env python3
"""Download and install pyqex from GitHub releases.

Automatically detects platform and Python version to download the correct wheel.

Usage:
    python scripts/install_pyqex.py [--version v1.0.0]
"""

import argparse
import platform
import subprocess
import sys
import tempfile
import urllib.request
import json
from pathlib import Path


REPO = "mfagerlund/pyqex"
GITHUB_API = f"https://api.github.com/repos/{REPO}/releases"


def get_platform_tag():
    """Get the wheel platform tag for the current system."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "win_amd64"
        else:
            return "win32"
    elif system == "darwin":
        if machine == "arm64":
            return "macosx_11_0_arm64"
        else:
            return "macosx_10_15_x86_64"
    elif system == "linux":
        if machine == "x86_64":
            return "manylinux_2_17_x86_64.manylinux2014_x86_64"
        elif machine == "aarch64":
            return "manylinux_2_17_aarch64.manylinux2014_aarch64"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def get_python_tag():
    """Get the wheel Python tag for the current interpreter."""
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def get_latest_release():
    """Get the latest release from GitHub."""
    url = f"{GITHUB_API}/latest"
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # No releases yet, try to list all
            with urllib.request.urlopen(GITHUB_API) as resp:
                releases = json.loads(resp.read().decode())
                if releases:
                    return releases[0]
        raise


def get_release_by_tag(tag):
    """Get a specific release by tag."""
    url = f"{GITHUB_API}/tags/{tag}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())


def find_matching_wheel(release, python_tag, platform_tag):
    """Find a wheel matching the current Python and platform."""
    assets = release.get("assets", [])

    # Try exact match first
    for asset in assets:
        name = asset["name"]
        if name.endswith(".whl"):
            if python_tag in name and platform_tag in name:
                return asset["browser_download_url"], name

    # Try partial platform match (for manylinux variations)
    platform_base = platform_tag.split(".")[0] if "." in platform_tag else platform_tag
    for asset in assets:
        name = asset["name"]
        if name.endswith(".whl"):
            if python_tag in name and platform_base in name:
                return asset["browser_download_url"], name

    return None, None


def download_and_install(url, filename):
    """Download wheel and install with pip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = Path(tmpdir) / filename
        print(f"Downloading {filename}...")

        with urllib.request.urlopen(url) as resp:
            wheel_path.write_bytes(resp.read())

        print(f"Installing {filename}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)
        ])

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Install pyqex from GitHub releases")
    parser.add_argument("--version", "-v", help="Specific version tag (e.g., v1.0.0)")
    parser.add_argument("--list", "-l", action="store_true", help="List available releases")
    args = parser.parse_args()

    if args.list:
        with urllib.request.urlopen(GITHUB_API) as resp:
            releases = json.loads(resp.read().decode())
            print("Available releases:")
            for r in releases[:10]:
                print(f"  {r['tag_name']}: {r['name']}")
        return

    python_tag = get_python_tag()
    platform_tag = get_platform_tag()
    print(f"Python: {python_tag}, Platform: {platform_tag}")

    try:
        if args.version:
            release = get_release_by_tag(args.version)
        else:
            release = get_latest_release()
    except urllib.error.HTTPError as e:
        print(f"Error fetching release: {e}")
        print("\nNo releases found. Build from source instead:")
        print("  git clone https://github.com/mfagerlund/pyqex.git")
        print("  cd pyqex && pip install .")
        sys.exit(1)

    print(f"Release: {release['tag_name']}")

    url, filename = find_matching_wheel(release, python_tag, platform_tag)

    if not url:
        print(f"\nNo wheel found for {python_tag} on {platform_tag}")
        print("Available wheels:")
        for asset in release.get("assets", []):
            if asset["name"].endswith(".whl"):
                print(f"  {asset['name']}")
        print("\nBuild from source instead:")
        print("  pip install git+https://github.com/mfagerlund/pyqex.git")
        sys.exit(1)

    download_and_install(url, filename)


if __name__ == "__main__":
    main()

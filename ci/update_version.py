#!/usr/bin/env python3
"""
Update version and commit information in ray_ascend/_version.py
"""

import argparse
import os
import subprocess
import sys


def get_git_commit():
    """Get the current git commit hash"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, text=True
        ).strip()
        return commit
    except subprocess.CalledProcessError as e:
        print(f"Error getting git commit: {e}", file=sys.stderr)
        return None


def get_git_tag():
    """Get the current git tag, or None if not on a tag"""
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
        # Remove leading 'v' prefix if present
        if tag.startswith("v"):
            tag = tag[1:]
        return tag
    except subprocess.CalledProcessError:
        # Not on a tag, return None
        return None


def update_version_file(version_file_path, version, commit):
    """Update the version file with new version and commit"""
    content = f'commit = "{commit}"\n'
    content += f'version = "{version}"\n\n'
    content += 'if __name__ == "__main__":\n'
    content += f'    print("%s %s" % (version, commit))\n'

    try:
        with open(version_file_path, "w") as f:
            f.write(content)
        print(f"Successfully updated {version_file_path}")
        print(f"Version: {version}")
        print(f"Commit: {commit}")
        return True
    except Exception as e:
        print(f"Error updating version file: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update version and commit information"
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to use (if not provided, will try to get from git tag)",
    )
    parser.add_argument(
        "--commit",
        help="Git commit hash to use (if not provided, will use current commit)",
    )
    parser.add_argument(
        "--file",
        default="ray_ascend/_version.py",
        help="Path to version file (default: ray_ascend/_version.py)",
    )

    args = parser.parse_args()

    # Get commit hash
    if args.commit:
        commit = args.commit
    else:
        commit = get_git_commit()
        if not commit:
            sys.exit(1)

    # Get version
    if args.version:
        version = args.version
    else:
        version = get_git_tag()
        if not version:
            print("Warning: Not on a git tag, using default version", file=sys.stderr)
            version = "0.1.0+dev"

    # Update version file
    version_file_path = os.path.abspath(args.file)
    if not update_version_file(version_file_path, version, commit):
        sys.exit(1)


if __name__ == "__main__":
    main()

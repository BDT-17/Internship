"""Clone or pull this repository inside Kaggle before running an entrypoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("/kaggle/working/Internship"))
    parser.add_argument("--branch", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.repo_dir / ".git").exists():
        run(["git", "fetch", "--all"], cwd=args.repo_dir)
        checkout_target = args.branch or "HEAD"
        run(["git", "checkout", checkout_target], cwd=args.repo_dir)
        run(["git", "pull", "--ff-only"], cwd=args.repo_dir)
    else:
        command = ["git", "clone"]
        if args.branch:
            command.extend(["--branch", args.branch])
        command.extend([args.repo_url, str(args.repo_dir)])
        run(command)

    print("Repository ready at:", args.repo_dir)


if __name__ == "__main__":
    main()


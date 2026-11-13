"""Create .env from .env.example (if missing) and print the next steps. Never touches secrets."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists():
        shutil.copy(example, env)
        print("Created .env from .env.example. Open it in Notepad and paste YOUR Hugging Face READ token after HF_TOKEN=")
    else:
        print(".env already exists (left unchanged).")
    print("Next: pip install -r requirements.txt   then   python scripts/validate_environment.py")


if __name__ == "__main__":
    main()

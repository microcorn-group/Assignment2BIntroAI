"""Deprecated tutorial script.

This repository now focuses on the ICS pipeline. Use `src/ics_ml.py` and
the CLI/eval tools instead. This file remains only as a tiny stub to avoid
confusion if referenced.
"""

if __name__ == "__main__":
    print("This tutorial script is deprecated. Train/evaluate with:")
    print("  python -m src.ics_eval --data-dir data --epochs 8")
    raise SystemExit(0)
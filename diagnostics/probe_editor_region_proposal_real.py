"""Run the integrated local region planner through the real editor manager.

Historical prototype artifacts remain in diagnostics/out. This runner uses
the current production guards and configuration without experimental overrides.
"""
import argparse
from diagnostics.probe_editor_bald_compliance import real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("photo")
    parser.add_argument("prompt")
    args = parser.parse_args()
    real(args.label, args.photo, args.prompt, "")


if __name__ == "__main__":
    main()

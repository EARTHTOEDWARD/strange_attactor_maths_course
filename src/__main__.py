"""
Command-line interface for Strange Attractor Math Course.

Usage:
    python -m src.visualise --demo
    python -m src.cheatsheet
"""

import sys
import argparse


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Strange Attractor Math Course CLI",
        epilog="For more information, see the notebooks/ directory"
    )
    
    parser.add_argument(
        "module",
        choices=["visualise", "cheatsheet", "maps"],
        help="Module to run"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (for visualise module)"
    )
    
    args = parser.parse_args()
    
    if args.module == "visualise":
        if args.demo:
            from .visualise import demo
            demo()
        else:
            print("Use --demo flag to run visualization demo")
            print("Example: python -m src.visualise --demo")
    
    elif args.module == "cheatsheet":
        from .cheatsheet import print_cheatsheet
        print_cheatsheet()
    
    elif args.module == "maps":
        print("Available dynamical systems:")
        from .maps import SYSTEMS
        for name, cls in SYSTEMS.items():
            system = cls()
            print(f"  - {name}: {system.description}")


if __name__ == "__main__":
    # Handle module-specific invocations
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        from .visualise import demo
        demo()
    else:
        main()
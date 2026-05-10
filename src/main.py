# main.py
# This executes the main workflow of the program.

import sys
from pathlib import Path

# Resolve path to workflow_service.py.
sys.path.insert(0, str(Path(__file__).parent / "services"))
from workflow_service import WorkflowService
from tkinter_app import GUI_App

def main() -> None:
    """Launch the Tkinter application."""
    workflow = WorkflowService()
    GUI_App(workflow)

if __name__ == "__main__":
    main()

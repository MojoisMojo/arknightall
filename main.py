import sys
import os
from PyQt6.QtWidgets import QApplication

# Ensure the 'src' directory is in the Python path
# This allows importing modules from 'src' when running main.py from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from ui.main_window import MainWindow
except ImportError as e:
    print("Error: Could not import MainWindow.")
    print("Ensure you are running this script from the project root directory (ArknightALL)")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    """
    Main function to initialize and run the PyQt application.
    """
    # Create the QApplication instance
    # Pass sys.argv to allow command-line arguments for Qt if needed
    app = QApplication(sys.argv)

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    # Start the Qt event loop
    # sys.exit() ensures that the application exits cleanly
    # and returns the appropriate exit code
    sys.exit(app.exec())

if __name__ == "__main__":
    # Check if the script is being run directly
    # (as opposed to being imported as a module)
    main()
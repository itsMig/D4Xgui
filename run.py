#!/usr/bin/env python3
"""
D4Xgui Application Runner

This script automatically sets up and runs the D4Xgui Streamlit application.
On first run, it creates a virtual environment and installs dependencies.
On subsequent runs, it simply starts the application.

Usage: python run.py
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path


class D4XguiRunner:
    """Handles the setup and execution of the D4Xgui application."""
    
    def __init__(self):
        self.app_dir = Path(__file__).parent.absolute()
        self.venv_dir = self.app_dir / "venv"
        self.requirements_file = self.app_dir / "requirements.txt"
        self.main_app = self.app_dir / "Welcome.py"
        self.is_windows = platform.system() == "Windows"
        
        # Virtual environment paths
        if self.is_windows:
            self.venv_python = self.venv_dir / "Scripts" / "python.exe"
            self.venv_pip = self.venv_dir / "Scripts" / "pip.exe"
            self.activate_script = self.venv_dir / "Scripts" / "activate.bat"
        else:
            self.venv_python = self.venv_dir / "bin" / "python"
            self.venv_pip = self.venv_dir / "bin" / "pip"
            self.activate_script = self.venv_dir / "bin" / "activate"
    
    def print_banner(self):
        """Print the application banner."""
        print("=" * 60)
        print("🧪 D4Xgui - Clumped Isotope Data Processing Tool")
        print("=" * 60)
        print()
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required.")
            print(f"   Current version: {sys.version}")
            print("   Please install a newer version of Python.")
            sys.exit(1)
        else:
            print(f"✅ Python version: {sys.version.split()[0]}")
    
    def check_requirements_file(self):
        """Check if requirements.txt exists."""
        if not self.requirements_file.exists():
            print("❌ Error: requirements.txt not found!")
            print("   Please ensure requirements.txt is in the same directory as run.py")
            sys.exit(1)
        else:
            print("✅ Requirements file found")
    
    def check_main_app(self):
        """Check if the main application file exists."""
        if not self.main_app.exists():
            print("❌ Error: Welcome.py not found!")
            print("   Please ensure Welcome.py is in the same directory as run.py")
            sys.exit(1)
        else:
            print("✅ Main application file found")
    
    def create_virtual_environment(self):
        """Create a virtual environment."""
        print("\n📦 Creating virtual environment...")
        try:
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_dir)
            ], check=True, capture_output=True, text=True)
            print("✅ Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating virtual environment: {e}")
            print("   Please ensure you have the 'venv' module available.")
            sys.exit(1)
    
    def upgrade_pip(self):
        """Upgrade pip in the virtual environment."""
        print("\n🔧 Upgrading pip...")
        try:
            subprocess.run([
                str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            print("✅ Pip upgraded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error upgrading pip: {e}")
            # Continue anyway, as this is not critical
    
    def install_requirements(self):
        """Install requirements from requirements.txt."""
        print("\n📚 Installing dependencies...")
        print("   This may take several minutes, please be patient...")
        
        try:
            # Install requirements with progress indication
            process = subprocess.Popen([
                str(self.venv_pip), "install", "-r", str(self.requirements_file)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            # Show some progress indication
            dots = 0
            while process.poll() is None:
                print("." * (dots % 4 + 1) + " " * (3 - dots % 4), end="\r")
                dots += 1
                time.sleep(0.5)
            
            if process.returncode == 0:
                print("✅ Dependencies installed successfully" + " " * 10)
            else:
                output = process.stdout.read()
                print(f"❌ Error installing dependencies:")
                print(output)
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            sys.exit(1)
    
    def is_venv_setup(self):
        """Check if virtual environment is already set up."""
        return (self.venv_dir.exists() and 
                self.venv_python.exists() and 
                self.activate_script.exists())
    
    def setup_environment(self):
        """Set up the virtual environment and install dependencies."""
        print("\n🚀 Setting up D4Xgui for the first time...")
        
        if not self.is_venv_setup():
            self.create_virtual_environment()
            self.upgrade_pip()
            self.install_requirements()
            
            # Create a marker file to indicate setup is complete
            setup_marker = self.venv_dir / ".setup_complete"
            setup_marker.touch()
            
            print("\n✅ Setup completed successfully!")
        else:
            print("✅ Virtual environment already exists")
    
    def test_application(self):
        """Testing app workflow."""
        print("\n🧑🏻‍🔬 Testing app workflow...")
        try:
            result = subprocess.run([
                str(self.venv_python), 'test_app_workflow.py'
            ], capture_output=True, text=True)
            results_file = self.app_dir / "test_assertion_results.txt"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    first_line = f.readline().strip()
                    if "ALL ASSERTIONS PASSED" in first_line:
                        print("✅ All tests passed successfully!")
                        return True
                    else:
                        print("⚠️  Some tests failed. Check test_assertion_results.txt for details")
                        # Print failed assertions from stdout
                        if result.stdout:
                            print("\nFailed assertions:")
                            print(result.stdout)
            
            if result.returncode == 0:
                print("✅ Test execution completed")
                return True
            else:
                print(f"❌ Tests failed with exit code: {result.returncode}")
                if result.stderr:
                    print(f"   Error output: {result.stderr}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running tests: {e}")
            if e.stderr:
                print(f"   Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during testing: {e}")
            return False        

    def run_application(self):
        """Run the Streamlit application."""
        print("\n🌐 Starting D4Xgui application...")
        print("   The application will open in your default web browser.")
        print("   Press Ctrl+C to stop the application.")
        print("\n" + "=" * 60)
        
        try:
            # Change to the app directory
            os.chdir(self.app_dir)
            
            # Run streamlit with the virtual environment's python
            subprocess.run([
                str(self.venv_python), "-m", "streamlit", "run", str(self.main_app),
                "--server.headless", "false",
                "--server.port", "8501",
                "--server.address", "localhost"
            ], check=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running application: {e}")
            print("   Please check that all dependencies are installed correctly.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            sys.exit(1)
    
    def run(self):
        """Main execution method."""
        self.print_banner()
        
        # Pre-flight checks
        self.check_python_version()
        self.check_requirements_file()
        self.check_main_app()
        
        # Setup environment if needed
        if not self.is_venv_setup():
            self.setup_environment()
            
        else:
            print("✅ Environment already set up")
        
        self.test_application()

        # Run the application
        self.run_application()


def main():
    """Main entry point."""
    try:
        runner = D4XguiRunner()
        runner.run()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
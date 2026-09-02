#!/usr/bin/env python3
"""
D4Xgui Application Runner

This script runs the D4Xgui Streamlit application.
When installed via pip/uvx, it uses the installed environment.
When run standalone, it can optionally set up a local venv.

Usage:
  - After pip install: D4Xgui
  - Standalone: python run.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

from tools import config as cfg


class D4XguiRunner:
	"""Handles the execution of the D4Xgui application."""
	
	def __init__(self):
		self.app_dir = Path(__file__).parent.absolute()
		self.main_app = self.app_dir / "Welcome.py"
		self.is_installed = self._check_if_installed()
	
	def _check_if_installed(self):
		"""Check if running from an installed package."""
		try:
			# Try to import streamlit to see if dependencies are available
			import streamlit
			return True
		except ImportError:
			return False
	
	def print_banner(self):
		"""Print the application banner."""
		print("=" * 60)
		print("🧪 D4Xgui - Clumped Isotope Data Processing Tool")
		print("=" * 60)
		print()
	
	def check_dependencies(self):
		"""Check if required dependencies are available."""
		missing_deps = []
		required = ['streamlit', 'pandas', 'numpy', 'plotly']
		
		for dep in required:
			try:
				__import__(dep)
			except ImportError:
				missing_deps.append(dep)
		
		if missing_deps:
			print("❌ Missing required dependencies:")
			for dep in missing_deps:
				print(f"   - {dep}")
			print("\n💡 Please install D4Xgui with:")
			print("   pip install D4Xgui")
			print("   or")
			print("   uvx D4Xgui")
			sys.exit(1)
		else:
			print("✅ All dependencies available")
	
	def check_main_app(self):
		"""Check if the main application file exists."""
		if not self.main_app.exists():
			print("❌ Error: Welcome.py not found!")
			print(f"   Expected at: {self.main_app}")
			sys.exit(1)
		else:
			print("✅ Main application file found")
	
	def run_application(self):
		"""Run the Streamlit application."""
		print("\n🌐 Starting D4Xgui application...")
		print("   The application will open in your default web browser.")
		print("   Press Ctrl+C to stop the application.")
		print("\n" + "=" * 60)
		
		try:
			# Change to the app directory
			os.chdir(self.app_dir)
			cfg.ensure_runtime()
			
			# Run streamlit with the current Python interpreter
			subprocess.run([
				sys.executable, "-m", "streamlit", "run", str(self.main_app),
				"--server.headless", "false",
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
		
		# Check dependencies
		self.check_dependencies()
		self.check_main_app()
		
		# Run the application
		self.run_application()


def main():
	"""Main entry point."""
	try:
		runner = D4XguiRunner()
		runner.run()
	except KeyboardInterrupt:
		print("\n\n👋 Interrupted by user.")
		sys.exit(0)
	except Exception as e:
		print(f"\n❌ Fatal error: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()

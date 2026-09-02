# Installation Guide

## Prerequisites

Before installing **D4Xgui**, ensure that your system meets the following requirements:

- **Python**: Version 3.11 or newer is required.
  - Check your version running: `python --version` or `python3 --version`.
  - Download from [python.org/downloads](https://www.python.org/downloads/) if needed.

---

## 🚀 Quick Start (Recommended)

The easiest way to run D4Xgui is using `uv`. This runs the application in an isolated environment without affecting your globally installed packages.

1. **Install uv** (if you haven't already):
   
   **macOS / Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Run D4Xgui:**
   
   ```bash
   uvx d4xgui
   ```
   
   The application will automatically download, install dependencies, and launch in your default web browser.
   After installation of D4Xgui via `uv`, you can find the project folder by prompting `uv cache dir`.

---

## 📦 Classic Installation

If you prefer to install D4Xgui permanently in your Python environment, follow the steps for your operating system below. We strongly recommend using a virtual environment.

### 🍎 macOS

1. Open **Terminal**.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install D4Xgui via pip:
   ```bash
   pip install D4Xgui
   ```
4. Run the application:
   ```bash
   D4Xgui
   ```

### 🐧 Linux (Ubuntu/Debian)

1. Open **Terminal**.
2. Ensure you have `pip` and `venv` installed:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install D4Xgui:
   ```bash
   pip install D4Xgui
   ```
5. Run the application:
   ```bash
   D4Xgui
   ```

### 🪟 Windows

1. Open **Command Prompt** or **PowerShell**.
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install D4Xgui:
   ```powershell
   pip install D4Xgui
   ```
4. Run the application:
   ```powershell
   D4Xgui
   ```

## Optional password
D4Xgui creates `.streamlit/secrets.toml` automatically on first run (from the shipped `secrets.toml.example`). Streamlit does **not** require this file; when it is missing or has no `password` key, the app runs without authentication.

To enable password protection, edit `.streamlit/secrets.toml` in the D4Xgui package folder (find it via `uv cache dir` after a `uvx` install) and add:

```toml
password = "YOURPW"
```

## User settings
Personal settings are stored in `user_settings.json` next to the app. This file is **not** shipped with releases. On a fresh install, built-in defaults from `tools/config.py` are used automatically; `user_settings.json` is created the first time you save changes on the **Settings** page.

## Optional port
If you want to change the preset port (1337), please open `.streamlit/config.toml` within the project folder (you can find the project folder by prompting `uv cache dir`) and modify the `server.port` parameter.




## Using the Application
Once running:
- The application opens automatically in your web browser
- Default address: `http://localhost:1337`
- If it doesn't open automatically, copy this address into your browser

## Stopping the Application
To stop D4Xgui:
- Press `Ctrl+C` in the terminal/command prompt window
- Or simply close the terminal/command prompt window


---

**Happy processing with D4Xgui! 🧪**

# D4Xgui - Easy Setup Guide for Users

Welcome! This guide will help you to set up and run D4Xgui.

## 📋 Prerequisites

You need Python installed on your computer. If you don't have Python:

### Windows:
1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python 3.12 or newer
3. **Important**: During installation, check "Add Python to PATH"

### macOS:
1. Go to [python.org](https://www.python.org/downloads/)
2. Download Python 3.12 or newer
3. Install the downloaded package

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

**Optional password:**
- If you want to protect D4Xgui with a password, please open `.streamlit/secrets.toml`and add a line `password=YOURPW`.


## 🚀 Running D4Xgui

### Option 1: Simple Double-Click (Recommended)

**Windows Users:**
- Double-click `run.bat`

**macOS/Linux Users:**
- Double-click `run.sh` (you may need to right-click and select "Open with Terminal")


### Option 2: Using Command Line

1. Open Terminal (macOS/Linux) or Command Prompt (Windows)
2. Change directory to the D4Xgui folder using `cd path/to/D4Xgui`
3. Run: `python run.py` (Windows) or `python3 run.py` (macOS/Linux)

## 🔄 What Happens on First Run

The first time you run D4Xgui:

1. ✅ **Environment Check**: Verifies Python version and required files
2. 📦 **Virtual Environment**: Creates an isolated Python environment
3. 📚 **Dependencies**: Installs all required packages (this takes a few minutes)
4. 🧑🏻‍🔬 **Testing**: Runs automated testing on example data
5. 🌐 **Launch**: Opens the application in your web browser

**Be patient during the first run** - installing dependencies can take a few minutes depending on your system and internet connection.

## 🔄 Subsequent Runs

After the first setup, running D4Xgui will:
- Skip the setup process
- Launch directly in your browser
- Start much faster

## 🌐 Using the Application

Once running:
- The application opens automatically in your web browser
- Default address: `http://localhost:8501`
- Default password is `123`
- If it doesn't open automatically, copy this address into your browser

## 🛑 Stopping the Application

To stop D4Xgui:
- Press `Ctrl+C` in the terminal/command prompt window
- Or simply close the terminal/command prompt window

## 🔧 Troubleshooting

### "Python not found" error:
- Make sure Python is installed and added to your system PATH
- Try using `python3` instead of `python`

### Permission errors on macOS/Linux:
```bash
chmod +x run.sh
./run.sh
```

### Dependencies installation fails:
- Check your internet connection
- Try running again - sometimes temporary network issues cause failures

### Port already in use:
- If you see "Port 8501 is already in use", close any other Streamlit applications
- Or wait a few minutes and try again

## 📁 File Structure

After setup, your folder will contain:
```
D4Xgui/
├── run.py              # Main launcher script
├── run.bat             # Windows launcher
├── run.sh              # macOS/Linux launcher
├── Welcome.py          # Main application
├── requirements.txt    # Dependencies list
├── venv/               # Virtual environment (created automatically)
├── pages/              # Application pages
└── tools/              # Application utilities
```

## 💡 Tips

- **Keep the terminal window open** while using the application
- **Don't delete the `venv` folder** - it contains all installed dependencies
- **Internet connection required** only for the first setup
- **The application runs locally** - your data stays on your computer

## 🆘 Need Help?

If you encounter issues:
1. Make sure Python 3.8+ is installed
2. Check that you have internet connection for the first run
3. Try running the setup again
4. Check the terminal output for specific error messages

---

**Happy processing with D4Xgui! 🧪**

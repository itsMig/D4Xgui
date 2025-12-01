__all__ = [
    "DataIOPage",
    "BaselineCorrectionPage",
    "ProcessingPage",
    "StandardizationResultsPage",
    "DualClumpedSpacePage",
    "DiscoverResultsPage",
    "D47CrunchPlotsPage",
    "SaveReloadPage",
    "DatabaseManagementPage",
]


import sys
import os
import importlib.util
import inspect
from unittest.mock import MagicMock

# Mock streamlit to avoid secrets errors during import
sys.modules['streamlit'] = MagicMock()

# Helper function to import module from file path

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Base directory for pages
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'D4Xgui','pages'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'D4Xgui')))
# Import modules dynamically by filename
modules = {}
modules['01_Data_IO'] = import_module_from_path('01_Data_IO', os.path.join(base_dir, '01_Data_IO.py'))
modules['03_Baseline_correction'] = import_module_from_path('03_Baseline_correction', os.path.join(base_dir, '03_Baseline_correction.py'))
modules['04_Processing'] = import_module_from_path('04_Processing', os.path.join(base_dir, '04_Processing.py'))
modules['05_Standardization_Results'] = import_module_from_path('05_Standardization_Results', os.path.join(base_dir, '05_Standardization_Results.py'))
modules['06_Dual_Clumped_Space'] = import_module_from_path('06_Dual_Clumped_Space', os.path.join(base_dir, '06_Dual_Clumped_Space.py'))
modules['07_Discover_Results'] = import_module_from_path('07_Discover_Results', os.path.join(base_dir, '07_Discover_Results.py'))
modules['08_D47crunch_plots'] = import_module_from_path('08_D47crunch_plots', os.path.join(base_dir, '08_D47crunch_plots.py'))
modules['100_Save_Reload'] = import_module_from_path('100_Save_Reload', os.path.join(base_dir, '100_Save_and_Reload.py'))
modules['97_Database_Management'] = import_module_from_path('97_Database_Management', os.path.join(base_dir, '97_Database_Management.py'))

# Re-export all public classes and functions from each module
__all__ = []
for mod_name, mod in modules.items():
    for name, obj in inspect.getmembers(mod):
        if not name.startswith('_') and (inspect.isclass(obj) or inspect.isfunction(obj)):
            # Set the __module__ attribute to the original module name
            if 'PlotParameters' in name:
                continue
            if getattr(obj, '__module__', None) != 'builtins':
                try:
                    obj.__module__ = f"pages.{mod_name}"
                except TypeError:
                    pass
            globals()[name] = obj
            __all__.append(name)

# Optionally, add any other public classes/functions from other modules here

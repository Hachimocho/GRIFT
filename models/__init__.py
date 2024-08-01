import os
import importlib
import inspect

# Get the directory of this __init__.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import all classes from all .py files in this directory
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        module = importlib.import_module(f'.{module_name}', package=__name__)
        
        # Get all classes from the module and add them to the global namespace
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                globals()[name] = obj

# Clean up temporary variables
del filename, module_name, module, name, obj, current_dir
import os
import importlib
import inspect
import sys

def import_classes_from_directory(directory):
    imported_classes = {}
    # Get the caller's global namespace
    caller_globals = sys._getframe(1).f_globals
    
    for filename in os.listdir(directory):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Remove .py extension
            module_path = f"{os.path.basename(directory)}.{module_name}"
            module = importlib.import_module(module_path)
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    imported_classes[name] = obj
                    # Add the class to the caller's global namespace
                    caller_globals[name] = obj
    
    return list(imported_classes.keys())

def get_classes_from_module(module_name):
    module = __import__(module_name)
    return [name for name, obj in module.__dict__.items() 
            if inspect.isclass(obj) and obj.__module__.startswith(module_name)]
    
def get_tagged_classes_from_module(module_name, tags):
    module = __import__(module_name)
    return [name for name, obj in module.__dict__.items() 
            if inspect.isclass(obj) and obj.__module__.startswith(module_name) and tags in obj.tags]
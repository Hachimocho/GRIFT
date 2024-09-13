import os
import importlib
import inspect
import sys
# from trainers import *
# from data import *
# from models import *
# from graphs import *
# from edges import *
# from nodes import *
# from managers import *
# from dataloaders import *
# from datasets import *
# from trainers import *
# from utils import *
# from traversals import *

def import_classes_from_directory(directory):
    """
    Import all classes from all .py files in the given directory and
    add them to the caller's global namespace. Returns a list of the
    names of the imported classes.
    
    :param directory: The directory to search for .py files.
    :type directory: str
    :return: A list of the names of the imported classes.
    :rtype: list[str]
    """
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
    """
    Retrieves a list of class names from a given module.

    Parameters:
    module_name (str): The name of the module to retrieve classes from.

    Returns:
    list: A list of class names from the module.
    """
    module = __import__(module_name)
    return [name for name, obj in module.__dict__.items() 
            if inspect.isclass(obj) and obj.__module__.startswith(module_name)]
    
def get_tagged_classes_from_module(module_name, tags):
    """
    Retrieves a list of class names from a given module that have a specific tag.

    Parameters:
    module_name (str): The name of the module to retrieve classes from.
    tags (list or str): A single tag or a list of tags to filter by.

    Returns:
    list: A list of class names from the module with the given tag(s).
    """
    module = __import__(module_name)
    return [name for name, obj in module.__dict__.items() 
            if inspect.isclass(obj) and obj.__module__.startswith(module_name) and tags in obj.tags]
    
    
def load_class_from_globals(params):
    """
    Takes a dictionary with one key and value as input, loads the key as a class name from globals, and passes it the value as parameters before returning it.

    Parameters:
    params (dict): A dictionary with one key and one value.

    Returns:
    object: An instance of the class specified by the key, with the value passed in as parameters.
    """
    assert len(params) == 1, "Dictionary must have one key and one value"
    name, args = next(iter(params.items()))
    clas = globals()[name]
    if isinstance(args, dict):
        args = {key: value for key, value in args.items()}
    return clas(**args)

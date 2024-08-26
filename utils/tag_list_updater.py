import sys
import os
from utils.import_utils import get_classes_from_module, import_classes_from_directory
from nodes import *
from edges import *
from data import *
from dataloaders import *
from datasets import *
from traversals import *
from models import *
from graphs import *
from managers import *

def update_tag_list():
    """
    Updates the tag list by retrieving the available classes from different modules
    and generating a dictionary of tags and the corresponding classes. The resulting
    dictionary is then written to a file named 'docs/tags.md' in the format:
    '# docs/tags.md\n\n'
    '`{tag}`: {", ".join(classes)}\n'

    Parameters:
    None

    Returns:
    None
    """
    # Get lists of available classes from different modules
    available_node_types = get_classes_from_module('nodes')  # Nodes
    available_edge_types = get_classes_from_module('edges')  # Edges
    available_data_types = get_classes_from_module('data')  # Data
    available_dataloader_types = get_classes_from_module('dataloaders')  # Dataloaders
    available_dataset_types = get_classes_from_module('datasets')  # Datasets
    available_traversal_types = get_classes_from_module('traversals')  # Traversals
    available_model_types = get_classes_from_module('models')  # Models
    available_graph_types = get_classes_from_module('graphs')  # Graphs
    available_manager_types = get_classes_from_module('managers')  # Managers

    # Combine all available class lists into a single list
    types = [
        available_node_types,
        available_edge_types,
        available_data_types,
        available_dataloader_types,
        available_dataset_types,
        available_traversal_types,
        available_model_types,
        available_graph_types,
        available_manager_types
    ]

    # Create a dictionary of tags and the corresponding classes
    tags = {}
    for type_list in types:
        for type_name in type_list:
            for tag in globals()[type_name].tags:
                if isinstance(tag, list):
                    tag = frozenset(tag)
                if tag in tags.keys():
                    tags[tag].append(type_name)
                else:
                    tags[tag] = [type_name]

    # Write the tags and classes to a file in markdown format
    with open('docs/tags.md', 'w') as f:
        f.write('# docs/tags.md\n\n')  # Start of markdown file
        for tag, classes in tags.items():
            if isinstance(tag, frozenset):
                tag = ", ".join(tag)
            f.write(f'`{tag}`: {", ".join(classes)}\n')  # Write each tag and corresponding classes

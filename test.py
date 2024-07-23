from utils.import_utils import import_classes_from_directory

# Usage
classes = import_classes_from_directory('./dataloaders')

# Get list of class names as strings
print("Imported classes:", classes)

d = DataLoader("./dataloaders")
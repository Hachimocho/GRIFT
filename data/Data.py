class Data():
    """
    Takes data from some source and converts it into a DataLoader-compatible format.
    Mostly in place so that wonky data type conversions can be implemented.
    Must have a tags attribute so other modules can define compatability with it.
    """
    
    tags: list[str] = ["all"]
    
    def __init__(self, indata):
        self.data = indata
        
    def load_data(self):
        return self.data
    
    def set_data(self, indata):
        self.data = indata
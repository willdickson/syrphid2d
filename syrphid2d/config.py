import toml

class Config(dict):

    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as f:
             data = toml.load(f)
        self.update(data)

    def print(self):
        print(toml.dumps(self))








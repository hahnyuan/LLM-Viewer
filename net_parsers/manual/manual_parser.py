
class ManualParser:
    def __init__(self):
        pass

    def parse(self,model_id,manual_params):
        with open(self.path, 'r') as file:
            return file.read()
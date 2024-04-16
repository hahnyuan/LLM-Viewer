
class BaseParser:
    def __init__(self, model_id, args):
        self.args = args

    def parse(self):
        raise NotImplementedError
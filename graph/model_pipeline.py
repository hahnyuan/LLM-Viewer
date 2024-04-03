
class ModelPipeline:
    def __init__(self):
        self.models = []

    def add_model(self, model: Model):
        self.models.append(model)

from scene.Gate import Gate
from scene.GateLarge import GateLarge

from src.python.samplegen.shotgen import OpenGlView


class GateGLThickLarge(OpenGlView):
    def __init__(self, model: Gate = GateLarge(), path: str = "resource/gates/", filename: str = "gate2016_large.obj"):
        super().__init__(model, path, filename)

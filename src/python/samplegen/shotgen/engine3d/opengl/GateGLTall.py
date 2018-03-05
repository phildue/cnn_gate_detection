from scene.Gate import Gate
from scene.GateTall import GateTall

from src.python.samplegen.shotgen import OpenGlView


class GateGLTall(OpenGlView):
    def __init__(self, model: Gate = GateTall(), path: str = "resource/gates/", filename: str = "gate2016.obj"):
        super().__init__(model, path, filename)

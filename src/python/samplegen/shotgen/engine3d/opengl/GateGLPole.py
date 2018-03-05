from scene.Gate import Gate
from scene.GateThin250 import Gate250

from src.python.samplegen.shotgen import OpenGlView


class GateGLPole(OpenGlView):
    def __init__(self, model: Gate = Gate250(), path: str = "resource/gates/", filename: str = "gate2016_bigpole.obj"):
        super().__init__(model, path, filename)

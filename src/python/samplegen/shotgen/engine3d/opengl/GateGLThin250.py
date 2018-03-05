from scene.Gate import Gate
from scene.GateThin250 import Gate250

from src.python.samplegen.shotgen import OpenGlView


class GateGLThin250(OpenGlView):
    def __init__(self, model: Gate = Gate250(), path: str = "resource/gates/", filename: str = "gate250.obj"):
        super().__init__(model, path, filename)

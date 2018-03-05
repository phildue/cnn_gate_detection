from scene.Gate import Gate
from scene.GateOpen import GateOpen

from src.python.samplegen.shotgen import OpenGlView


class GateGLOpen(OpenGlView):
    def __init__(self, model: Gate = GateOpen(), path: str = "resource/gates/", filename: str = "open_gate.obj"):
        super().__init__(model, path, filename)

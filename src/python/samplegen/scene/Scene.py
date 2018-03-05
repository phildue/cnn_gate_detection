from scene import Light
from scene.Camera import Camera

from src.python.samplegen.shotgen import View
from src.python.utils.labels.Pose import Pose


class Scene:
    def __init__(self, cam: Camera = Camera(1000), objects: [(View, Pose)] = None, lights: [Light] = None):
        self.lights = lights
        self.cam = cam
        self.objects = objects

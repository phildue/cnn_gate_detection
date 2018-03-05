from scene.Light import Light

from src.python.samplegen.shotgen import LightGen


class ConstantLightGen(LightGen):
    def __init__(self, lights: [Light]):
        self.lights = lights

    def gen_lights(self):
        return self.lights

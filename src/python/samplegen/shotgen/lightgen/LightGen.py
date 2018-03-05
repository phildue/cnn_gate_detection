from abc import ABC, abstractmethod

from src.python.samplegen.scene import Light


class LightGen(ABC):
    @abstractmethod
    def gen_lights(self) -> [Light]:
        pass

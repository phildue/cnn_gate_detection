from abc import ABC, abstractmethod

from src.python.samplegen.scene import GateCorners


class Gate(ABC):
    @property
    @abstractmethod
    def corners(self) -> GateCorners:
        pass

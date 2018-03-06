from abc import ABC, abstractmethod

from samplegen.scene.GateCorners import GateCorners


class Gate(ABC):
    @property
    @abstractmethod
    def corners(self) -> GateCorners:
        pass

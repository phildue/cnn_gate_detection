from time import sleep

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.shotgen.positiongen.PositionGen import PositionGen
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AirSimRecord:

    def __init__(self, airsim: AirSimClient):
        self.airsim = airsim

    def generate(self, n_samples) -> [(Image, ImgLabel)]:
        samples = []
        labels = []
        for i in range(n_samples):
            img, label = self.airsim.retrieve_samples()
            samples.append(img)
            labels.append(label)
            sleep(0.1)

        return samples,labels

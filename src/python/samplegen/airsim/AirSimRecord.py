from time import sleep

from samplegen.airsim.AirSimClient import AirSimClient
from samplegen.shotgen.positiongen.PositionGen import PositionGen
from utils.imageprocessing.Image import Image
from utils.labels.ImgLabel import ImgLabel


class AirSimRecord:

    def __init__(self, airsim: AirSimClient, period=0.1):
        self.period = period
        self.airsim = airsim

    def generate(self, n_samples) -> [(Image, ImgLabel)]:
        samples = []
        labels = []
        responses = []

        for i in range(n_samples):
            response = self.airsim.query_airsim()
            responses.append(response)
            sleep(self.period)

        for r in responses:
            img, label = self.airsim.response2sample(r)
            samples.append(img)
            labels.append(label)

        return samples, labels

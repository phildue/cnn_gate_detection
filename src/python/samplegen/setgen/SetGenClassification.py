from imggen.RandomImgGen import RandomImgGen

from src.python.samplegen.shotgen import ShotGen


class SetGenClassification:
    def __init__(self, shotgen: ShotGen, imggen: RandomImgGen):
        self.imggen = imggen
        self.shotgen = shotgen

    def get_samples(self, n_shots, n_backgrounds):
        shots, labels = self.shotgen.get_shots(n_shots)
        gate_samples, gate_labels = self.imggen.generate(shots, labels, n_backgrounds)
        nogate_samples, nogate_labels = self.imggen.gen_empty_samples(n_backgrounds)
        return {
            'gate': (gate_samples, gate_labels),
            'nogate': (nogate_samples, nogate_labels)
        }

    def get_set(self, n_shots_train, n_shots_test, n_backgrounds_train: int, n_backgrounds_test: int):
        return {
            'training': self.get_samples(n_shots_train, n_backgrounds_train),
            'test': self.get_samples(n_shots_test, n_backgrounds_test)
        }

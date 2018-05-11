from pprint import pprint

import numpy as np

from modelzoo.evaluation.ModelEvaluator import ModelEvaluator
from modelzoo.models.Predictor import Predictor
from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import save_file
from utils.timing import toc, tic
import keras.backend as K


class SpeedEvaluator(ModelEvaluator):
    def __init__(self, model: Predictor, out_file=None, verbose=True):
        super().__init__(model, out_file, verbose)
        self.predict = K.function([model.net.backend.input, K.learning_phase()], [model.net.backend.output])

    def evaluate_generator(self, generator: DatasetGenerator, n_batches=10):
        it = iter(generator.generate())
        results = []
        for i in range(n_batches):
            batch = next(it)
            images = [b[0] for b in batch]
            tic()
            sample_t = self.model.preprocessor.preprocess_batch(images)
            time_enc = toc()
            tic()
            netout = self.predict([sample_t, 0])[0]
            time_pred = toc()

            tic()
            predictions = self.model.postprocessor.postprocess(netout)
            time_pp = toc()
            fps_enc = len(images) / time_enc
            fps_pred = len(images) / time_pred
            fps_pp = len(images) / time_pp

            fps_total = len(images) / (time_enc + time_pred + time_pp)

            results_batch = {'T_Encoding [s]': 1 / fps_enc,
                             'T_Prediction [s]': 1 / fps_pred,
                             'T_Postprocessing [s]': 1 / fps_pp,
                             'T_Total [s]': 1 / fps_total}

            results.append(results_batch)

            if self.verbose:
                pprint("Evaluated batch {0:d}/{1:d}".format(i, n_batches))
                pprint(results_batch)
            if self.out_file is not None:
                content = {'results': results}
                save_file(content, self.out_file)

        return results

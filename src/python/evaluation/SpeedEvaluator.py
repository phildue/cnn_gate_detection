from pprint import pprint

import keras.backend as K
import numpy as np
from modelzoo.models.Predictor import Predictor

from utils.fileaccess.DatasetGenerator import DatasetGenerator
from utils.fileaccess.utils import save_file
from utils.timing import toc, tic


class SpeedEvaluator:
    def __init__(self, model: Predictor, out_file=None, verbose=True):
        self.verbose = verbose
        self.out_file = out_file
        self.predict = K.function([model.net.backend.input, K.learning_phase()], [model.net.backend.output])

    def evaluate_generator(self, generator: DatasetGenerator, n_batches=10):
        it = iter(generator.generate())
        results_enc = np.zeros((n_batches,))
        results_pred = np.zeros((n_batches,))
        results_pp = np.zeros((n_batches,))
        results_total = np.zeros((n_batches,))
        for i in range(n_batches):
            batch = next(it)
            images = [b[0] for b in batch]
            tic()
            sample_t = self.model.preprocessor.preprocess_batch(images)
            time_enc = toc(verbose=False)
            tic()
            netout = self.predict([sample_t, 0])[0]
            time_pred = toc(verbose=False)

            tic()
            predictions = self.model.postprocessor.postprocess(netout)
            time_pp = toc(verbose=False)

            results_enc[i] = time_enc / len(images)
            results_pred[i] = time_pred / len(images)
            results_pp[i] = time_pp / len(images)
            results_total[i] = (time_enc + time_pred + time_pp) / len(images)

        results_dict = {'T_Encoding [s]': np.mean(results_enc),
                        'T_Prediction [s]': np.mean(results_pred),
                        'T_Postprocessing [s]': np.mean(results_pp),
                        'T_Total [s]': np.mean(results_total)}
        if self.verbose:
            pprint("Done.")
            pprint(results_dict)
        if self.out_file is not None:
            content = {'results_enc': results_enc,
                       'results_pred': results_pred,
                       'results_pp': results_pp,
                       'results_mean': results_dict}
            save_file(content, self.out_file)

        return results_dict

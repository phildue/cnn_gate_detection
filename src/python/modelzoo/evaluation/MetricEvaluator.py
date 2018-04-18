from modelzoo.evaluation.Metric import Metric
from utils.fileaccess.utils import load_file, save_file
from utils.imageprocessing.Backend import imread
from utils.labels.ImgLabel import ImgLabel
from utils.timing import toc, tic


class MetricEvaluator:
    def __init__(self, metrics: [Metric], color_format='bgr', out_file=None, verbose=True):
        self.verbose = verbose
        self.out_file = out_file
        self.metrics = {m.__class__.__name__: m for m in metrics}
        self.color_format = color_format

    def evaluate_sample(self, label_pred: ImgLabel, label_true: ImgLabel, img=None):

        results = {}
        for name, m in self.metrics.items():
            m.update(label_true=label_true, label_pred=label_pred)

            results[name] = m.result

            if m.show and img is not None:
                m.show_result(img)

        return results

    def evaluate(self, labels_true: [ImgLabel], labels_pred: [ImgLabel], image_files: [str]):
        results = {m: [] for m in self.metrics.keys()}
        tic()
        for i in range(len(labels_true)):
            image = imread(image_files[i], self.color_format)
            sample_result = self.evaluate_sample(labels_pred[j], labels_true[j], image)
            for m in self.metrics.keys():
                results[m].append(sample_result[m])

        if self.verbose:
            toc("Evaluated file in ")

        if self.out_file is not None:
            content = {'results': results,
                       'labels_true': labels_true,
                       'labels_pred': labels_pred}
            save_file(content, self.out_file)

        return results

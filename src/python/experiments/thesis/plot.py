import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.utils import sum_results, average_precision_recall
from utils.fileaccess.utils import load_file


def plot_result(models: [str], names: [str], n_iterations=1, ious=None, work_dir='out/thesis/', validation_set=False):
    if ious is None:
        ious = [0.4, 0.6, 0.8]
    simset = 'iros2018_course_final_simple_17gates'
    realsets = [
        'jevois_cyberzoo',
        'jevois_basement',
        'jevois_hallway',
    ]
    frame = pd.DataFrame()
    frame['Name'] = pd.Series(names)

    '''
    Validation mAP
    '''
    if validation_set:
        for i_iou, iou in enumerate(ious):
            aps = []
            errs = []
            for m, model in enumerate(models):
                total_detections = []
                for i in range(n_iterations):
                    model_dir = model + '_i0{}'.format(i)
                    result_file = work_dir + model_dir + '/test_valid/' + 'results_iou{}.pkl'.format(iou)
                    try:
                        results = load_file(result_file)
                        total_detections.append(sum_results(results['results']))
                    except FileNotFoundError:
                        print("Not found: {}".format(model_dir))

                m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
                meanAp = np.mean(m_p)
                errAp = np.mean(std_p)
                aps.append(np.round(meanAp, 2))  # , errAp
                errs.append(np.round(errAp, 2))
            frame['Val Data' + str(iou)] = pd.Series(aps)
            frame['Val Data' + str(iou)+' Err'] = pd.Series(errs)

    plt.figure(figsize=(8, 3))
    for i_iou, iou in enumerate(ious):
        plt.subplot(1, len(ious), i_iou + 1)
        plt.title('VE IoU:{}'.format(iou))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title("Results in ")
        plt.ylim(0.0, 1.1)
        aps = []
        errs = []
        for m, model in enumerate(models):
            total_detections = []
            for i in range(n_iterations):
                model_dir = model + '_i0{}'.format(i)
                result_file = work_dir + model_dir + '/test_' + simset + '/' + 'results_iou{}.pkl'.format(iou)
                try:
                    results = load_file(result_file)
                    total_detections.append(sum_results(results['results']))
                except FileNotFoundError:
                    print("Not found: {}".format(model_dir))

            m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
            meanAp = np.mean(m_p)
            errAp = np.mean(std_p)
            aps.append(np.round(meanAp, 2))  # , errAp
            errs.append(np.round(errAp, 2))
            plt.errorbar(m_r, m_p, std_p)
        frame['Sim Data' + str(iou)] = pd.Series(aps)
        frame['Sim Data' + str(iou) + ' Err'] = pd.Series(errs)

    plt.figure(figsize=(8, 3))

    for i_iou, iou in enumerate(ious):
        plt.subplot(1, len(ious), i_iou + 1)
        plt.title('RW IoU:{}'.format(iou))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.1)
        aps = []
        errs = []
        for m, model in enumerate(models):
            total_detections = []
            for i in range(n_iterations):
                detections_set = []
                for j, d in enumerate(realsets):
                    model_dir = model + '_i0{}'.format(i)
                    result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou)
                    if "snake" in model:
                        result_file = work_dir + model + '{}_boxes{}-{}_iou{}_i0{}.pkl'.format(d, 0, 2.0, iou, i)
                    try:
                        results = load_file(result_file)
                        detections_set.append(sum_results(results['results']))
                    except FileNotFoundError:
                        print("Not Found: {}".format(model_dir))
                if len(detections_set) > 0:
                    total_detections.append(sum_results(detections_set))
            m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
            meanAp = np.mean(m_p, 0)
            errAP = np.mean(std_p, 0)
            plt.errorbar(m_r, m_p, std_p)
            aps.append(np.round(meanAp, 2))  # , errAp
            errs.append(np.round(errAP, 2))

        frame['Real Data' + str(iou)] = pd.Series(aps)
        frame['Real Data' + str(iou) + ' Err'] = pd.Series(errs)

    frame.set_index('Name')
    plt.legend(names)

    plt.figure(figsize=(8, 3))

    w = 1 / len(models)
    w -= w * 0.1
    subplots = 3 if validation_set else 2
    subplots_idx = 1

    if validation_set:
        plt.subplot(1, subplots, subplots_idx)
        subplots_idx += 1
        plt.title('Training Set', fontsize=12)
        for i, m in enumerate(models):
            bars = []
            errs = []
            for iou in ious:
                bars.append(frame['Val Data' + str(iou)][i])
                errs.append(frame['Val Data' + str(iou) + ' Err'][i])

            plt.bar(np.arange(len(ious)) - w + i * w,
                    bars, width=w, yerr=errs)
            plt.xticks(np.arange(len(ious)), ious)
            plt.xlabel('Intersection Over Union')
            plt.ylabel('Average Precision')
            plt.ylim(0, 0.9)

    plt.subplot(1, subplots, subplots_idx)
    subplots_idx += 1
    plt.title('Simulated Data', fontsize=12)
    for i, m in enumerate(models):
        bars = []
        errs = []
        for iou in ious:
            bars.append(frame['Sim Data' + str(iou)][i])
            errs.append(frame['Sim Data' + str(iou) + ' Err'][i])

        plt.bar(np.arange(len(ious)) - w + i * w,
                bars, width=w, yerr=errs)
        plt.xticks(np.arange(len(ious)), ious)
        plt.xlabel('Intersection Over Union')
        plt.ylabel('Average Precision')
        plt.ylim(0, 0.9)

    plt.subplot(1, subplots, subplots_idx)
    subplots_idx += 1
    plt.title('Real Data', fontsize=12)
    for i, m in enumerate(models):
        bars = []
        errs = []
        for iou in ious:
            bars.append(frame['Real Data' + str(iou)][i])
            errs.append(frame['Real Data' + str(iou) + ' Err'][i])

        plt.bar(np.arange(len(ious)) - w + i * w,
                bars, width=w, yerr=errs)
        plt.xticks(np.arange(len(ious)), ious)
        plt.xlabel('Intersection Over Union')
        plt.ylabel('Average Precision')
        plt.ylim(0, 0.9)

    plt.legend(names, bbox_to_anchor=(1.1, 1.05))
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                        wspace=0.3, hspace=0.3)
    return frame

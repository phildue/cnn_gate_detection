import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelzoo.evaluation.utils import average_precision_recall, sum_results
from utils.fileaccess.utils import load_file
from utils.workdir import cd_work

cd_work()
models = [
    'yolov3_gate_varioussim416x416',
    'yolov3_gate_dronemodel416x416',
    'yolov3_allview416x416',
    # 'yolov3_blur416x416',
    # 'yolov3_chromatic416x416',
]

work_dir = 'out/thesis/datagen/'
n_iterations = 1

names = [
    'Random View Points',
    'Flight',
    'Combined'
    # 'Flight + Random Blur',
    # 'Flight + Random Chrom',
]
testset = 'iros2018_course_final_simple_17gates'
plt.figure(figsize=(8, 3))
plt.title('Precision - Recall IoU:{}'.format(0.6))
plt.subplot(1, 2, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Results in Virtual Environment")
frame = pd.DataFrame()
frame['Name'] = pd.Series(names)
plt.ylim(0.0, 1.1)
ious = [0.4, 0.6, 0.8]
for iou in ious:
    ap = []
    recall = []
    for model in models:
        total_detections = []
        mean_detections = []
        for i in range(n_iterations):
            # model_dir = model + '_i0{}'.format(i)
            # result_file = work_dir + model_dir + '/test_' + testset + '/' + 'predictions.pkl'.format(iou)
            # if "snake" in model:
            #     result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou, i)
            # # try:
            # results = evalmetric('view', work_dir+ model_dir, result_file, min_box_area=0.01, max_box_area=2.0,min_aspect_ratio=0.33,max_aspect_ratio=3.0,
            #                      iou_thresh=iou)
            model_dir = model + '_i0{}'.format(i)
            result_file = work_dir + model_dir + '/test_' + testset + '/' + 'results_iou{}.pkl'.format(iou)
            if "snake" in model:
                result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou, i)
            try:
                results = load_file(result_file)
                total_detections.append(sum_results(results['results']))
            except FileNotFoundError:
                continue
        recall.append(total_detections[0].recall(0.5))
        m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
        print(m_p)
        print('{}:  map{}: {}'.format(model, iou, np.mean(m_p)))
        if iou == 0.6:
            plt.plot(m_r, m_p,'x--')
            # plt.plot(total_detections[0].recall_conf, np.linspace(0, 1, 11), 'x--')
            # plt.plot(total_detections[0].precision_conf, np.linspace(0, 1, 11), 'x--')
        ap.append(np.mean(m_p))
    frame['Sim Data' + str(iou)] = pd.Series(ap)
    frame['Sim Data Recall' + str(iou)] = pd.Series(recall)
plt.legend(names)

datasets = [
    'jevois_cyberzoo',
    'jevois_basement',
    'jevois_hallway',
]
datasets_names = [
    'Cyberzoo',
    'Basement',
    'Hallway'
]

plt.subplot(1, 2, 2)
plt.title('Results on Real World Datasets'.format(0.6))
plt.xlabel('Recall')
plt.ylabel('Precision')
#
plt.ylim(0.0, 1.1)

for iou in ious:
    ap = []
    recall = []
    for model in models:
        precision = np.zeros((len(datasets), 11))
        recalls = np.zeros((len(datasets), 11))
        r = 0
        for j, d in enumerate(datasets):
            total_detections = []
            for i in range(n_iterations):
                # model_dir = model + '_i0{}'.format(i)
                # result_file = work_dir + model_dir + '/test_' + d + '/' + 'predictions.pkl'.format(iou)
                # if "snake" in model:
                #     result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou, i)
                # # try:
                # results = evalmetric('view', work_dir + model_dir, result_file, min_box_area=0.01, max_box_area=2.0,
                #                      iou_thresh=iou)
                model_dir = model + '_i0{}'.format(i)
                result_file = work_dir + model_dir + '/test_' + d + '/' + 'results_iou{}.pkl'.format(iou)
                if "snake" in model:
                    result_file = 'out/thesis/snake/test_{}_results_iou{}_{}.pkl'.format(testset, iou, i)
                try:
                    results = load_file(result_file)
                    total_detections.append(sum_results(results['results']))
                except FileNotFoundError:
                    continue
            r += total_detections[0].recall(0.5) / len(datasets)
            m_p, m_r, std_p, std_R = average_precision_recall(total_detections)
            precision[j] = m_p
            recalls[j] = m_r
        if iou == 0.6:
            plt.plot(np.mean(recalls, 0), np.mean(precision, 0), 'x--')
        meanAp = np.mean(precision, 0)
        recall.append(r)
        ap.append(np.round(np.mean(meanAp), 2))  # , np.mean(np.mean(err_p, 0))
    frame['Real Data' + str(iou)] = pd.Series(ap)
    frame['Real Data Recall' + str(iou)] = pd.Series(recall)

plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/view_pr.png')
plt.legend(names)
print(frame.to_string())

w = 1 / len(models)
w -= w * 0.1
plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.title('Simulated Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(ious)) - w + i * w,
            [frame['Sim Data0.4'][i], frame['Sim Data0.6'][i], frame['Sim Data0.8'][i]], width=w)
    plt.xticks(np.arange(len(ious)), ious)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Average Precision')
    plt.ylim(0, 0.8)

plt.subplot(1, 2, 2)
plt.title('Real Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(ious)) - w + i * w,
            [frame['Real Data0.4'][i], frame['Real Data0.6'][i], frame['Real Data0.8'][i]], width=w)
    plt.xticks(np.arange(len(ious)), ious)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Average Precision')
    plt.ylim(0, 0.8)

plt.legend(names, bbox_to_anchor=(1.1, 1.05))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/view_bar.png')

plt.figure(figsize=(8, 3))

plt.subplot(1, 2, 1)
plt.title('Simulated Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(ious)) - w + i * w,
            [frame['Sim Data Recall0.4'][i], frame['Sim Data Recall0.6'][i], frame['Sim Data Recall0.8'][i]], width=w)
    plt.xticks(np.arange(len(ious)), ious)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Recall')
    plt.ylim(0, 0.8)

plt.subplot(1, 2, 2)
plt.title('Real Data', fontsize=12)
for i, m in enumerate(models):
    plt.bar(np.arange(len(ious)) - w + i * w,
            [frame['Real Data Recall0.4'][i], frame['Real Data Recall0.6'][i], frame['Real Data Recall0.8'][i]],
            width=w)
    plt.xticks(np.arange(len(ious)), ious)
    plt.xlabel('Intersection Over Union')
    plt.ylabel('Recall')
    plt.ylim(0, 0.8)

plt.legend(names, bbox_to_anchor=(1.1, 1.05))
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None,
                    wspace=0.3, hspace=0.3)
plt.savefig('doc/thesis/fig/recall_bar.png')

plt.show(True)

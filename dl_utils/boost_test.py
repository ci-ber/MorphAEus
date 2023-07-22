"""
boost_test.py

Offline script to run evaluation and statistics using test bootstrapping
"""
from dl_utils import get_data_from_csv
from dl_utils import set_seed
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from optim.metrics import *
import torchvision.transforms as transforms
from transforms.preprocessing import AddChannelIfNeeded, AssertChannelFirst, ReadImage, To01
import matplotlib.pyplot as plt


dataset_id = 0
dataset_keys = ['AE-S', 'AE-D', 'VAE', 'b-VAE', 'AAE', 'SI-VAE', 'DAE', 'MorphAEus']
################  0       1       2       3       4       5        6        7

predictions_path = './path/to/weights/cxr/' ## TODO: set path
predictions_path += (dataset_keys[dataset_id] + '/residuals/')

basic_path = './path/to/library/' ## TODO: set path
normal_csvs = ['/data/CXR/splits/cxr_normal_test.csv', '/data/padchest/splits/cxr_normal_test.csv']
pathology_csvs = ['/data/CXR/splits/cxr_opa_test.csv', '/data/CXR/splits/cxr_covid_padchest_test.csv']
normal_paths = ['Normal_RSNA', 'Normal_Padchest']
pathology_paths = ['Pneumonia', 'Covid']

seeds = [101822, 101922, 102022, 102122, 102222]

TPIL = transforms.ToPILImage()
TT = transforms.ToTensor()
RES = transforms.Resize(128)
Gray = transforms.Grayscale()

image_t = transforms.Compose([ReadImage(), To01(), AddChannelIfNeeded(), TPIL, RES, TT])

def compute_residual(img, prediction):
    img_r = img
    rec_r = prediction
    res = np.abs(img_r - rec_r)
    return res, img_r, rec_r

average_auroc = 0
for idx_task in range(len(normal_paths)):
    print(f'^^^^^^^^^^^^^^^^ TASK {pathology_paths[idx_task]} - {dataset_keys[dataset_id]}^^^^^^^^^^^^^^^^^^^^')
    normal_filename = normal_paths[idx_task]
    path_filename = pathology_paths[idx_task]
    normal_csv = basic_path + normal_csvs[idx_task]
    pathology_csv = basic_path + pathology_csvs[idx_task]
    normal_files = get_data_from_csv(normal_csv)
    pathology_files = get_data_from_csv(pathology_csv)
    predictions = []
    labels = []

    # Get predictions
    for normal_idx in tqdm.tqdm(range(len(normal_files))):
        img = image_t(basic_path + normal_files[normal_idx][1:])[0].numpy()
        x_rec = np.load(predictions_path + normal_filename + '_' + str(normal_idx) + '_res.npy')
        x_res, img_r, rec_r = compute_residual(img, x_rec)
        predictions.append(np.nanmean(x_res))
        labels.append(0)

    for pathology_idx in tqdm.tqdm(range(len(pathology_files))):
        img = image_t(basic_path + pathology_files[pathology_idx][1:])[0].numpy()
        x_rec = np.load(predictions_path + path_filename + '_' + str(pathology_idx) + '_res.npy')
        x_res, img_r, rec_r = compute_residual(img, x_rec)
        predictions.append(np.nanmean(x_res))
        labels.append(1)

    predictions_all = np.asarray(predictions)
    labels_all = np.asarray(labels)
    aurocs = []
    # 5 runs with different seeds
    for i in range(len(seeds)):
        set_seed(seeds[i])
        predictions_idx = np.random.choice(range(len(predictions)), size=len(predictions), replace=True)
        predictions = predictions_all[predictions_idx]
        labels = labels_all[predictions_idx]

        auprc = average_precision_score(labels, predictions)
        # print('[ {} ]: AUPRC: {}'.format(dataset_keys[dataset_id], auprc))
        auroc = roc_auc_score(labels, predictions)
        # print('[ {} ]: AUROC: {}'.format(dataset_keys[dataset_id], auroc))
        aurocs.append(auroc)

        fpr, tpr, ths = roc_curve(labels, predictions)
        th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
        th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
        fpr95 = fpr[th_95]
        fpr99 = fpr[th_99]
        # print('[ {} ]: FPR95: {} at th: {}'.format(dataset_keys[dataset_id], fpr95, ths[th_95]))
        # print('[ {} ]: FPR99: {} at th: {}'.format(dataset_keys[dataset_id], fpr99, ths[th_99]))

    print(f'^^^^^^^^^^^^^^^^ Final statistcs for {dataset_keys[dataset_id]}: AUROC:  {np.mean(aurocs)} +/- {np.std(aurocs)} ^^^^^^^^^^^^^^^^^^^^')
    average_auroc += np.mean(aurocs)  # Average of 5 runs
average_auroc /= 2  # Average of Pneumonia and Covid-19
print(f'Average AUROC: {average_auroc}')
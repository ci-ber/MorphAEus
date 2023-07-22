import logging
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
#
from torch.nn import L1Loss
#
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
#
import lpips
import pytorch_fid.fid_score as fid
#
from dl_utils import *
from optim.metrics import *
from core.DownstreamEvaluator import DownstreamEvaluator
from dl_utils.visualization import plot_warped_grid
import subprocess
import os


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.auprc = AUPRC()
        self.compute_fid = False
        self.compute_scores = True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """

        self.anomaly_detection(global_model)

    def anomaly_detection(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ ANOMALY DETECTION TEST #################")
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        path_mednist = './data/MedNIST_FID/'
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        batch_size = 1000
        dims = 2048
        num_workers = 4
        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'FID_CXR': [],
            'FID_OD': [],
        }
        pred_dict = dict()
        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x)

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_res.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_res = np.abs(x_rec_ - x_)

                    res_pred = np.mean(x_res)
                    label = 0 if 'Normal' in dataset_key else 1
                    pred_.append(res_pred)
                    label_.append(label)
                    if self.compute_fid:
                        im = Image.fromarray((x_rec_ * 255).astype(np.uint8))
                        if not os.path.exists(self.image_path + dataset_key):
                            os.makedirs(self.image_path + dataset_key)
                        im.save(self.image_path + dataset_key + '/' + dataset_key + str(res_pred) + '_' + str(count) + '.jpeg')
                    if (idx % 500) == 0 and (i % 5 == 0):
                        elements = [x_, x_rec_, x_res]
                        v_maxs = [1, 1, 0.5]
                        if 'x_prior' in x_rec_dict.keys():
                            global_prior = x_rec_dict['x_prior'].detach().cpu()[i].numpy()
                            elements = [x_, global_prior, x_rec_, np.abs(global_prior - x_), x_res]
                            v_maxs = [1, 1, 1, 0.5, 0.5]
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'inferno'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})

                        if 'x_prior' in x_rec_dict.keys():
                            deformation = x_rec_dict['deformation'][i].cpu().detach().numpy()
                            f, ax = plt.subplots(1, 1)
                            grid_img = plot_warped_grid(ax, deformation,
                                                        interval=1, color=(0.721, 0.450, 0.619, 0.75), linewidth=1)
                            wandb.log({"Grid_flow/" + '_' + dataset_key + '_' + str(count) + '_' +
                                       str(ssim_): [wandb.Image(grid_img, caption="Sample_" + str(count))]})

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

            # Compute FIDs
            if self.compute_fid:
                output_path = self.image_path + dataset_key + '/'
                fid_cxr = fid.calculate_fid_given_paths([output_path, path_mednist+'CXR'], batch_size, device, dims,
                                                        num_workers)
                metrics['FID_CXR'].append([fid_cxr])
                logging.info('[ {} ]: FID_CXR: {}'.format(dataset_key, fid_cxr))
                if dataset_key != 'Normal':
                    fid_od = fid.calculate_fid_given_paths([output_path, path_mednist+dataset_key], batch_size, device, dims,
                                                            num_workers)
                else:
                    fid_od = 0
                metrics['FID_OD'].append([fid_od])
                logging.info('[ {} ]: FID_OD: {}'.format(dataset_key, fid_od))
                logging.info('Starting MedNIST Classifier....')
                subprocess.call(["python", "./projects/classifier/trainer.py",
                                 "--classifier", "densenet121", "--pretrained", "1", "--test_phase", "1",
                                 "--num_classes", "5", "--csv_file", "False", "--eval_dir",
                                output_path])

        if self.compute_scores:
            normal_key = 'Normal'
            for key in pred_dict.keys():
                if 'Normal' in key:
                    normal_key = key
                    break
            pred_cxr, label_cxr = pred_dict[normal_key]
            for dataset_key in self.test_data_dict.keys():
                print(f'Running evaluation for {dataset_key}')
                if dataset_key == normal_key:
                    continue
                pred_ood, label_ood = pred_dict[dataset_key]
                predictions = np.asarray(pred_cxr + pred_ood)
                labels = np.asarray(label_cxr + label_ood)
                print('Negative Classes: {}'.format(len(np.argwhere(labels == 0))))
                print('Positive Classes: {}'.format(len(np.argwhere(labels == 1))))
                print('total Classes: {}'.format(len(labels)))
                print('Shapes {} {} '.format(labels.shape, predictions.shape))

                auprc = average_precision_score(labels, predictions)
                print('[ {} ]: AUPRC: {}'.format(dataset_key, auprc))
                auroc = roc_auc_score(labels, predictions)
                print('[ {} ]: AUROC: {}'.format(dataset_key, auroc))

                fpr, tpr, ths = roc_curve(labels, predictions)
                th_95 = np.squeeze(np.argwhere(tpr >= 0.95)[0])
                th_99 = np.squeeze(np.argwhere(tpr >= 0.99)[0])
                fpr95 = fpr[th_95]
                fpr99 = fpr[th_99]
                print('[ {} ]: FPR95: {} at th: {}'.format(dataset_key, fpr95, ths[th_95]))
                print('[ {} ]: FPR99: {} at th: {}'.format(dataset_key, fpr99, ths[th_99]))

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})
![morphaeus_logo](https://github.com/ci-ber/MorphAEus/assets/106509806/107b2aff-be67-4d4b-801d-015b1c3a884e width="200" height="200")

<h1 align="center">
  <br>
What Do AEs Learn? Challenging Common Assumptions in Unsupervised Anomaly Detection
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">MICCAI 2023</h4>
<h4 align="center"><a href="https://ci.bercea.net/project/morphaeus/">Project Website</a> • <a href="https://arxiv.org/pdf/2206.03698.pdf">Preprint</a> </h4>

![morphaeus_overview](https://github.com/ci-ber/MorphAEus/assets/106509806/9265fb3b-6d69-4f8b-ad73-4ff65f2a1813)


## Citation

If you find our work useful, please cite our paper:
```
@article{bercea2022we,
  title={What do we learn? debunking the myth of unsupervised outlier detection},
  author={Bercea, Cosmin I and Rueckert, Daniel and Schnabel, Julia A},
  journal={arXiv preprint arXiv:2206.03698},
  year={2022}
}
```

> **Abstract:** *Detecting abnormal findings in medical images is a critical task that enables timely diagnoses, effective screening, and urgent case prioritization. Autoencoders (AEs) have emerged as a popular choice for anomaly detection and have achieved state-of-the-art (SOTA) performance in detecting pathology. However, their effectiveness is often hindered by the assumption that the learned manifold only contains information that is important for describing samples within the training distribution. In this work, we challenge this assumption and investigate what AEs actually learn when they are posed to solve anomaly detection tasks. We have found that standard, variational, and recent adversarial AEs are generally not well-suited for pathology detection tasks where the distributions of normal and abnormal strongly overlap. In this work, we propose MorphAEus, novel deformable AEs to produce pseudo-healthy reconstructions refined by estimated dense deformation fields. Our approach improves the learned representations, leading to more accurate reconstructions, reduced false positives and precise localization of pathology. We extensively validate our method on two public datasets and demonstrate SOTA performance in detecting pneumonia and COVID-19.*


## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

#### Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### Clone repository

```bash
git clone https://github.com/ci-ber/MorphAEus.git
cd autoDDPM
```

#### Install requirements

```bash
pip install -r requirements.txt
```

#### Download datasets 

<h4 align="center"><a href="https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018">RSNA</a> • <a href="https://bimcv.cipf.es/bimcv-projects/padchest/">PadChest</a> </h4>

> *Alternatively you can use your own chest X-Ray images with our <a href=""> pre-trained weights</a> or train from scratch on other anatomies and modalities.*


#### Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/morphaeus/morphaeus.yaml
```

Refer to morphaeus.yaml for the default configuration. Store the pre-trained model from <a href=""> HERE</a> into the specified directory to skip the training part.







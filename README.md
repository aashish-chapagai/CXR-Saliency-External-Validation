# Can saliency maps flag chest radiograph classifier errors at a new hospital?

Code for *Can saliency maps flag chest radiograph classifier errors at a new hospital? An external validation study*
by Aashish Chapagai, Angad Singh Gurtatta and Erik Cambria. The archived version, with the trained models, is at
https://doi.org/10.5281/zenodo.23618485.

## Contents
| File | What it does |
|---|---|
| `1_Data_Preprocessing_and_EDA.ipynb` | Cleans NIH ChestX-ray14, splits it by patient, writes 320 × 320 arrays |
| `2_CNN_Training.ipynb` | Trains DenseNet-121, tunes thresholds, evaluates on the NIH test set |
| `3_Feature_Extraction_and_Classical_ML.ipynb` | Random forest and linear SVM on DenseNet-121 features |
| `4_ViT_Training.ipynb` | Trains ViT-B/16 (Google Colab, NVIDIA A100, Keras 3.15.1, KerasHub 0.21.1) |
| `5_Anatomy_Segmentation.ipynb` | Lung and heart masks for CheXpert and VinDr-CXR with HybridGNet |
| `6_External_Data_Preparation.ipynb` | Images, labels, annotations and thoracic regions for the three sites |
| `7_XAI_Analysis.ipynb` | Grad-CAM, LIME, SHAP, enrichment, localization and failure detection |
| `paper_figures.py` | Figures 1–5, Extended Data Figs 1–6 and the exploratory analyses |
| `results/` | The aggregate results reported in the paper |

## Data
No data are included. Download each dataset under its own terms and place it relative to this folder:
| Path | Contents | Source |
|---|---|---|
| `images/` | NIH ChestX-ray14 images | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| `Dataset.csv`, `BBox.csv` | `Data_Entry_2017_v2020.csv` and `BBox_List_2017.csv`, renamed | same |
| `external/chexlocalize/` | CheXpert validation and test images with CheXlocalize annotations | https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c |
| `external/chexmask/` | `ChestX-Ray8.csv`, `CheXpert.csv`, `VinDr-CXR.csv` | https://doi.org/10.13026/3705-zg36 |
| `external/vindr/` | VinDr-CXR test DICOMs, `annotations_test.csv`, `image_labels_test.csv` | https://doi.org/10.13026/3akn-b287 (credentialed) |
| `external/hybridgnet/` | `models/`, `utils/` and `weights/weights.pt` | https://huggingface.co/spaces/ngaggion/Chest-x-ray-HybridGNet-Segmentation |

## Environments
Step 5 uses the HybridGNet environment; all other steps use the main one.

    conda create -n cxr python=3.12 -y && conda activate cxr && pip install -r requirements.txt
    conda create -n hybridgnet python=3.12 -y && conda activate hybridgnet && pip install -r requirements-hybridgnet.txt

## Running
From this folder: steps 1–6, then step 7 twice (`MODEL = 'densenet'`,
then `MODEL = 'vit'`, in the third code cell), then `python paper_figures.py`. Outputs go to `models/` and `figures/`.
To reuse our models instead of training, put the Zenodo model files in `models/` and run steps 1, 5, 6 and 7 and
`paper_figures.py`.

## Results and data use
`results/` holds copies of the aggregate files the code writes to `models/`. Per-radiograph outputs are not included,
because those derived from CheXpert and VinDr-CXR cannot be redistributed. ChestX-ray14 was provided by the NIH
Clinical Center (Wang et al., CVPR 2017).

## Licence
MIT (see LICENSE).

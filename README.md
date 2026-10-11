# Glioma MRI Segmentation Using Deep Learning

> **Status: Work in Progress — Data Exploration and Initial Preprocessing Validation Completed**

The project has completed initial exploration of the BraTS 2024 glioma MRI dataset and developed a reproducible preprocessing pipeline using PyTorch and MONAI. Training and validation DataLoaders have been tested successfully on representative cases, and preliminary anatomical quality-control checks have been performed.

**Next stage:** Dataset-wide quality control and development of a 3D U-Net segmentation model.

No segmentation model has been trained or evaluated at this stage.

---

## Clinical Motivation

Gliomas are primary brain tumours with considerable heterogeneity in their imaging appearances and patterns of infiltration.

MRI plays a central role in their assessment. Different MRI sequences provide complementary information regarding tumour anatomy, contrast enhancement,
tissue water content, and surrounding signal abnormality.

Accurate segmentation of tumour-associated abnormalities and postoperative resection cavities can support quantitative assessment of disease burden, treatment-related anatomical changes, and longitudinal imaging findings. It also provides a foundation for computer-assisted neuro-oncological imaging analysis.

This project investigates how modern deep-learning techniques can be applied to this problem.

---

## Project Objectives

The project will progressively develop an end-to-end medical image segmentation workflow:

1. Explore and understand volumetric brain MRI data stored in NIfTI format.
2. Examine the complementary information provided by T1, contrast-enhanced T1, T2, and FLAIR MRI.
3. Develop a reproducible preprocessing pipeline for multiparametric MRI.
4. Train a deep-learning model for three-dimensional glioma segmentation.
5. Evaluate segmentation performance using appropriate quantitative metrics.
6. Visualize predicted segmentations alongside the original MRI and reference annotations.
7. Examine model errors and clinically relevant failure cases.

The initial modelling approach will use a **3D U-Net-based architecture**, implemented using PyTorch and MONAI.

---

## Dataset

This project uses the **BraTS 2024 Adult Glioma (BraTS-GLI) training dataset**, obtained through the BraTS 2024 Challenge on Synapse.

The dataset includes multimodal brain MRI and expert-derived segmentation annotations, including postoperative tumour-associated abnormalities and resection cavities.

The current dataset comprises **1,350 imaging cases from 613 subjects**, with four MRI modalities and one segmentation volume per case.

The original imaging data are not included in this GitHub repository. They are stored separately and excluded from Git version control.

**Dataset source:** [BraTS 2024 GLI Training Data — Synapse](https://www.synapse.org/Synapse:syn60086071)

**Dataset version:** V2 (18 June 2024), incorporating corrections to segmentation annotations documented in the dataset changelog.

---

## MRI Modalities

The dataset contains complementary MRI sequences used for tumour assessment:

| Sequence | Principal contribution |
| --- | --- |
| T1 | Anatomical and structural information |
| T1ce | Demonstration of contrast-enhancing tumour |
| T2 | Sensitive to increased tissue water content |
| FLAIR | Demonstrates T2-hyperintense abnormality while suppressing CSF signal |

These sequences can be combined as multiple imaging channels for the
segmentation model.

---

## MRI Preprocessing Pipeline

A reproducible preprocessing workflow has been developed using MONAI and PyTorch to prepare multimodal MRI volumes and reference segmentations for three-dimensional deep learning.

The pipeline includes:

- **Image loading:** Reading multimodal MRI and segmentation data from NIfTI files.
- **Spatial standardisation:** Converting images to a consistent RAS anatomical orientation.
- **Intensity normalisation:** Normalising nonzero MRI intensities separately for each modality.
- **Foreground cropping:** Removing unnecessary background while retaining the relevant anatomy.
- **Training augmentation:** Random 96 × 96 × 96 voxel patch sampling, spatial flipping and intensity scaling.
- **Segmentation conversion:** Converting categorical annotations into binary region masks.
- **Data loading:** Preparing batched image and segmentation tensors for model training and validation.

### Segmentation Regions

The preprocessing pipeline converts the supplied categorical segmentation annotations into four binary region masks:

| Region | Description |
| --- | --- |
| WT | Whole tumour-associated region |
| TC | Tumour core |
| ET | Enhancing tumour |
| RC | Resection cavity |

The regions are represented as separate channels for subsequent deep-learning segmentation.

The masks are not necessarily mutually exclusive: some tumour regions are nested within larger regions, while the resection cavity is represented separately.

The model will be developed to predict these anatomical regions from the four MRI modalities.

### Initial Validation Results

| Component | Observed output |
| --- | --- |
| Training MRI batch | `(2, 4, 96, 96, 96)` |
| Training segmentation batch | `(2, 4, 96, 96, 96)` |
| Validation MRI batch | `(1, 4, 133, 185, 132)` |
| Validation segmentation batch | `(1, 4, 133, 185, 132)` |

The validation dimensions above refer to one representative case; foreground-cropped dimensions can vary between cases.

Preliminary visual quality control demonstrated plausible alignment between MRI anatomy and the reference segmentation. Automated consistency checks also confirmed preservation of annotated foreground voxels following segmentation-label conversion.

These findings establish initial pipeline functionality, not comprehensive dataset validation or model performance.

For implementation details, see [`02_preprocessing_pipeline.ipynb`](notebooks/02_preprocessing_pipeline.ipynb).

---

## Planned Workflow

```text
BraTS MRI Data
      │
      ▼
Data Exploration
      │
      ▼
Preprocessing
      │
      ▼
3D MRI Volumes
(T1 / T1ce / T2 / FLAIR)
      │
      ▼
3D U-Net
      │
      ▼
Tumour and Postoperative Region Segmentation
      │
      ▼
Quantitative Evaluation
      │
      ▼
Visualization & Failure Analysis
```

---

## Repository Structure

```text
glioma-mri-segmentation/
│
├── data/
│   ├── raw/            # Dataset excluded from Git
│   ├── interim/        # Intermediate processing outputs
│   └── processed/      # Model-ready data
│
├── notebooks/
│   ├── 01_mri_data_exploration.ipynb
│   └── 02_preprocessing_pipeline.ipynb
│
├── src/                # Reusable project code
├── models/             # Saved model weights (excluded from Git)
├── outputs/
│   └── figures/        # Figures for analysis and documentation
│
├── README.md
└── .gitignore
```

The large source MRI dataset is stored outside the repository to prevent
accidental inclusion in GitHub.

---

## Current Progress

### Phase 1 — MRI Data Exploration (Completed)

- [x] Establish the project repository and Python environment.
- [x] Configure PyTorch, MONAI, NiBabel and CUDA support.
- [x] Download and organise the BraTS 2024 GLI training dataset.
- [x] Examine the NIfTI image format and MRI volume dimensions.
- [x] Explore T1, contrast-enhanced T1, T2 and T2-FLAIR MRI sequences.
- [x] Inspect categorical segmentation annotations and their spatial relationship to MRI anatomy.
- [x] Visualise MRI slices and segmentation overlays.

**Notebook:** [`01_mri_data_exploration.ipynb`](notebooks/01_mri_data_exploration.ipynb)

### Phase 2 — MRI Preprocessing (Initial Validation Completed)

- [x] Establish subject-level training, validation and test splits.
- [x] Implement consistent anatomical orientation and intensity normalisation.
- [x] Implement foreground cropping.
- [x] Implement random 3D patch sampling and training augmentation.
- [x] Develop reusable segmentation-label conversion transforms.
- [x] Test training and validation DataLoaders.
- [x] Verify tensor dimensions, numerical values and segmentation consistency.
- [x] Perform preliminary multimodal MRI and segmentation quality control.
- [ ] Conduct dataset-wide quality-control checks.
- [ ] Finalise reusable preprocessing components for model training.

**Notebook:** [`02_preprocessing_pipeline.ipynb`](notebooks/02_preprocessing_pipeline.ipynb)

### Phase 3 — Model Development (Planned)

- [ ] Implement a 3D U-Net architecture.
- [ ] Configure segmentation loss functions and evaluation metrics.
- [ ] Establish training and validation procedures.
- [ ] Train the segmentation model.
- [ ] Save model checkpoints and document experiments.

### Phase 4 — Model Evaluation (Planned)

- [ ] Evaluate region-specific Dice scores.
- [ ] Assess segmentation performance on held-out data.
- [ ] Visualise predicted segmentations alongside reference annotations.
- [ ] Investigate failure cases and model limitations.
- [ ] Summarise clinical relevance and potential applications.

**Current milestone:** The initial preprocessing pipeline is functioning on representative cases. Model development has not yet commenced.

---

## Future Directions

Following development and evaluation of the core segmentation pipeline, this project may be extended to investigate how automated tumour segmentation and
quantitative imaging can contribute to broader neuro-oncological applications.

### 1. 3D Reconstruction and Printing

One planned extension is to convert three-dimensional tumour segmentations into physical models suitable for 3D printing.

The proposed workflow is:

```text
Brain MRI
    │
    ▼
Deep-Learning Segmentation
    │
    ▼
3D Tumour Mask
    │
    ▼
Surface Reconstruction
    │
    ▼
3D Mesh
(STL / 3MF)
    │
    ▼
Mesh Processing
    │
    ▼
Slicing
    │
    ▼
3D-Printed Model
```

This extension would explore the complete computational pathway from medical imaging data to a physical representation of patient anatomy.

Potential areas of investigation include:

- conversion of voxel-based segmentation masks into surface meshes;
- mesh smoothing and geometric processing;
- generation of printable STL or 3MF models;
- representation of tumour and surrounding anatomical structures;
- assessment of how segmentation errors propagate into reconstructed models;
- exploration of physical models for anatomical visualization, education, and
  communication.

This extension is deliberately separated from the core segmentation project. The segmentation model must first be quantitatively evaluated before its output
is used for downstream three-dimensional reconstruction.

### 2. Spatial Prediction of Glioma Recurrence

A longer-term research direction is to extend tumour segmentation towards prediction of the spatial patterns of glioma recurrence and progression.

Glioblastoma is an infiltrative disease in which tumour cells may extend beyond abnormalities visible on conventional MRI. Of particular interest is whether
information from the visible tumour, surrounding tissue, and cerebral white-matter architecture can be combined to identify preferential pathways of
subsequent progression.

Recent research has investigated **diffusion tensor imaging (DTI)**, tractography, and computational modelling of tumour–white-matter relationships
as potential approaches for studying these pathways. Machine-learning and deep-learning methods provide additional means of extracting and integrating
multimodal spatial features.

A potential future workflow would be:

```text
Initial / Post-Treatment MRI
              │
              ▼
     3D Tumour Segmentation
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
Tumour & Margin    DTI / White-Matter
   Features          Tractography
       │             │
       └──────┬──────┘
              ▼
      Tumour–Tract Spatial
          Relationships
              │
              ▼
      Quantitative Features
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
 Machine Learning   Computational /
  / Deep Learning   Spatial Modelling
       │             │
       └──────┬──────┘
              ▼
       Predicted Spatial
       Recurrence-Risk Map
              │
              ▼
        Follow-up MRI
              │
              ▼
     Registration of Observed
          Recurrence
              │
              ▼
     Spatial Validation Against
        Actual Progression
```

Potential areas of investigation include:

- diffusion-derived imaging biomarkers such as fractional anisotropy and diffusivity measures;
- relationships between tumour boundaries and adjacent white-matter tracts;
- preferential directions of tumour infiltration;
- geometric and radiomic features of tumour margins and peritumoural tissue;
- tractography and structural connectivity;
- integration of multimodal imaging features using machine-learning or deep-learning methods;
- computational modelling of tumour spread along white-matter architecture;
- generation of interpretable spatial recurrence-risk maps;
- registration of longitudinal imaging to map observed recurrence; and
- comparison between predicted and observed spatial patterns of progression.

A particularly interesting extension would be to investigate whether detailed **three-dimensional tumour segmentation** provides useful spatial information
for recurrence modelling beyond tumour localization alone.

This represents a progression from asking:

> **Where is the tumour now?**

to investigating:

> **Where is the tumour likely to progress or recur?**

The BraTS dataset used for the current segmentation project does not contain the DTI data required for this work. Such an investigation would therefore
constitute a separate future research project requiring an appropriate longitudinal dataset containing diffusion imaging, follow-up imaging, and
suitable recurrence or progression outcomes.

These approaches remain investigational. Particular attention would be required to distinguish biologically meaningful predictive performance from patterns
arising from small datasets, synthetic augmentation, institutional imaging protocols, or assumptions incorporated into computational models.

Any future predictive model would therefore require rigorous evaluation using real longitudinal recurrence data, appropriate validation methodology, and,
where possible, independent or external datasets before conclusions regarding clinical utility could be drawn.

### 3. Additional Model Development

Other possible extensions include:

- comparison of alternative medical image segmentation architectures;
- evaluation of model generalizability across independent imaging datasets;
- uncertainty estimation and identification of low-confidence predictions;
- automated calculation of tumour volumes from segmentation masks;
- longitudinal comparison of tumour volumes and spatial patterns;
- extraction of quantitative imaging and radiomic features;
- integration of imaging features with clinical and molecular variables;
- development of clinician-oriented visualization tools for reviewing model
  predictions.

---

## Technology

The project currently uses:

- **Python**
- **PyTorch**
- **MONAI**
- **NiBabel**
- **NumPy**
- **Matplotlib**
- **Jupyter**
- **CUDA-enabled NVIDIA GPU acceleration**

Additional tools for 3D reconstruction, mesh processing, advanced imaging analysis, and spatial modelling may be incorporated as the project develops.

---

## Project Philosophy

The objective of this project is not simply to train a segmentation model.

The workflow is being developed incrementally to demonstrate understanding of the complete medical-imaging machine-learning pipeline: data representation,
preprocessing, model development, evaluation, visualization, and critical interpretation of model performance.

Particular emphasis will be placed on presenting results in a form that is understandable to clinicians as well as technically reproducible.

The longer-term aim is to explore how computational outputs can be translated beyond model performance metrics into forms that are meaningful for clinical,
educational, research, and physical applications.

The current segmentation project therefore also provides a technical foundation for potential future work involving three-dimensional anatomical reconstruction,
advanced MRI, longitudinal imaging, and spatial modelling of tumour progression.

---

## Project Status

This repository is under active development.

The commit history documents the project as it progresses from initial MRI exploration through preprocessing, model development, evaluation, and clinical
interpretation.

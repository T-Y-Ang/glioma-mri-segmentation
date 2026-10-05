# Glioma MRI Segmentation Using Deep Learning

> **Status: Work in Progress**

This project explores the use of deep learning for automated segmentation of
gliomas on multiparametric brain MRI.

It forms part of a broader portfolio exploring applications of machine learning
and artificial intelligence in neuroscience and neurosurgery.

---

## Clinical Motivation

Gliomas are primary brain tumours with considerable heterogeneity in their
imaging appearances and patterns of infiltration.

MRI plays a central role in their assessment. Different MRI sequences provide
complementary information regarding tumour anatomy, contrast enhancement,
tissue water content, and surrounding signal abnormality.

Accurate segmentation of tumour-associated regions can support quantitative
assessment of tumour burden and provides an important foundation for
computer-assisted neuro-oncological imaging analysis.

This project investigates how modern deep-learning techniques can be applied
to this problem.

---

## Project Objectives

The project will progressively develop an end-to-end medical image segmentation
workflow:

1. Explore and understand volumetric brain MRI data stored in NIfTI format.
2. Examine the complementary information provided by T1, contrast-enhanced T1,
   T2, and FLAIR MRI.
3. Develop a reproducible preprocessing pipeline for multiparametric MRI.
4. Train a deep-learning model for three-dimensional glioma segmentation.
5. Evaluate segmentation performance using appropriate quantitative metrics.
6. Visualize predicted segmentations alongside the original MRI and reference
   annotations.
7. Examine model errors and clinically relevant failure cases.

The initial modelling approach will use a **3D U-Net-based architecture**,
implemented using PyTorch and MONAI.

---

## Dataset

This project uses data from the **BraTS 2024 Adult Glioma dataset**.

BraTS provides multimodal brain MRI together with expert-derived tumour
segmentation annotations for the development and evaluation of computational
brain tumour analysis methods.

The imaging dataset is **not included in this repository**.

Large medical imaging files are stored separately from the source code and are
excluded from Git version control.

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
Tumour Segmentation
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
│   └── 01_mri_data_exploration.ipynb
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

### Phase 1 — MRI Data Exploration

- [x] Create project repository and directory structure
- [x] Configure dedicated Python environment
- [x] Configure GPU-enabled PyTorch
- [x] Install MONAI and NiBabel
- [x] Obtain access to the BraTS training dataset
- [x] Create initial MRI exploration notebook
- [ ] Load and inspect a representative NIfTI MRI volume
- [ ] Visualize axial, coronal, and sagittal views
- [ ] Compare T1, T1ce, T2, and FLAIR sequences
- [ ] Inspect tumour segmentation annotations
- [ ] Overlay segmentation masks on MRI

### Subsequent Phases

- [ ] MRI preprocessing
- [ ] Dataset preparation and train/validation/test strategy
- [ ] 3D U-Net implementation
- [ ] Model training
- [ ] Quantitative segmentation evaluation
- [ ] Prediction visualization
- [ ] Failure-case analysis
- [ ] Final clinical and technical interpretation

---

## Future Directions

Following development and evaluation of the core segmentation pipeline, this
project may be extended to investigate how automated tumour segmentation and
quantitative imaging can contribute to broader neuro-oncological applications.

### 1. 3D Reconstruction and Printing

One planned extension is to convert three-dimensional tumour segmentations into
physical models suitable for 3D printing.

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

This extension would explore the complete computational pathway from medical
imaging data to a physical representation of patient anatomy.

Potential areas of investigation include:

- conversion of voxel-based segmentation masks into surface meshes;
- mesh smoothing and geometric processing;
- generation of printable STL or 3MF models;
- representation of tumour and surrounding anatomical structures;
- assessment of how segmentation errors propagate into reconstructed models;
- exploration of physical models for anatomical visualization, education, and
  communication.

This extension is deliberately separated from the core segmentation project.
The segmentation model must first be quantitatively evaluated before its output
is used for downstream three-dimensional reconstruction.

### 2. Spatial Prediction of Glioma Recurrence

A longer-term research direction is to extend tumour segmentation towards
prediction of the spatial patterns of glioma recurrence and progression.

Glioblastoma is an infiltrative disease in which tumour cells may extend beyond
abnormalities visible on conventional MRI. Of particular interest is whether
information from the visible tumour, surrounding tissue, and cerebral
white-matter architecture can be combined to identify preferential pathways of
subsequent progression.

Recent research has investigated **diffusion tensor imaging (DTI)**,
tractography, and computational modelling of tumour–white-matter relationships
as potential approaches for studying these pathways. Machine-learning and
deep-learning methods provide additional means of extracting and integrating
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

- diffusion-derived imaging biomarkers such as fractional anisotropy and
  diffusivity measures;
- relationships between tumour boundaries and adjacent white-matter tracts;
- preferential directions of tumour infiltration;
- geometric and radiomic features of tumour margins and peritumoural tissue;
- tractography and structural connectivity;
- integration of multimodal imaging features using machine-learning or
  deep-learning methods;
- computational modelling of tumour spread along white-matter architecture;
- generation of interpretable spatial recurrence-risk maps;
- registration of longitudinal imaging to map observed recurrence; and
- comparison between predicted and observed spatial patterns of progression.

A particularly interesting extension would be to investigate whether detailed
**three-dimensional tumour segmentation** provides useful spatial information
for recurrence modelling beyond tumour localization alone.

This represents a progression from asking:

> **Where is the tumour now?**

to investigating:

> **Where is the tumour likely to progress or recur?**

The BraTS dataset used for the current segmentation project does not contain the
DTI data required for this work. Such an investigation would therefore
constitute a separate future research project requiring an appropriate
longitudinal dataset containing diffusion imaging, follow-up imaging, and
suitable recurrence or progression outcomes.

These approaches remain investigational. Particular attention would be required
to distinguish biologically meaningful predictive performance from patterns
arising from small datasets, synthetic augmentation, institutional imaging
protocols, or assumptions incorporated into computational models.

Any future predictive model would therefore require rigorous evaluation using
real longitudinal recurrence data, appropriate validation methodology, and,
where possible, independent or external datasets before conclusions regarding
clinical utility could be drawn.

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

Additional tools for 3D reconstruction, mesh processing, advanced imaging
analysis, and spatial modelling may be incorporated as the project develops.

---

## Project Philosophy

The objective of this project is not simply to train a segmentation model.

The workflow is being developed incrementally to demonstrate understanding of
the complete medical-imaging machine-learning pipeline: data representation,
preprocessing, model development, evaluation, visualization, and critical
interpretation of model performance.

Particular emphasis will be placed on presenting results in a form that is
understandable to clinicians as well as technically reproducible.

The longer-term aim is to explore how computational outputs can be translated
beyond model performance metrics into forms that are meaningful for clinical,
educational, research, and physical applications.

The current segmentation project therefore also provides a technical foundation
for potential future work involving three-dimensional anatomical reconstruction,
advanced MRI, longitudinal imaging, and spatial modelling of tumour progression.

---

## Project Status

This repository is under active development.

The commit history documents the project as it progresses from initial MRI
exploration through preprocessing, model development, evaluation, and clinical
interpretation.

## Current Progress

### Completed

- Set up a reproducible Python environment with GPU-enabled PyTorch, MONAI,
  NiBabel and supporting scientific Python libraries.
- Downloaded and verified the BraTS 2024 Adult Glioma training dataset.
- Extracted and validated 1,350 training cases containing 6,750 NIfTI volumes.
- Explored the structure of 3D NIfTI MRI data, including:
  - voxel arrays and image dimensions;
  - voxel spacing;
  - affine transformations;
  - anatomical orientation.
- Loaded and compared the four aligned MRI modalities:
  - native T1-weighted (`t1n`);
  - contrast-enhanced T1-weighted (`t1c`);
  - T2-weighted (`t2w`);
  - T2-FLAIR (`t2f`).
- Verified multimodal spatial alignment using image dimensions and NIfTI affine
  transformations.
- Loaded and explored the supplied reference tumour segmentations.
- Developed segmentation-driven selection of tumour-containing MRI slices.
- Visualised reference tumour masks over the corresponding MRI.
- Quantified labelled tumour compartments using physical voxel dimensions.

### In Progress

- MRI intensity analysis and normalization.
- Development of the preprocessing pipeline for 3D deep-learning input.

### Next Steps

- Construct model-ready multimodal MRI and segmentation tensors.
- Define reproducible training, validation and test splits.
- Implement a baseline 3D segmentation network.
- Train the model using the BraTS reference segmentations.
- Evaluate segmentation performance using quantitative metrics such as Dice
  similarity.
- Visualise model predictions against reference segmentations and analyse
  failure cases.
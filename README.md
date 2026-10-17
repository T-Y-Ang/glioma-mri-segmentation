# Glioma MRI Segmentation Using Deep Learning

**3D U-Net | BraTS 2024 GLI | PyTorch | MONAI**

**Project status:** Experimental research and independent held-out evaluation completed. Final repository documentation and reproducibility preparation in progress.

## Project Summary

This project investigates the application of deep learning to postoperative glioma MRI segmentation, with particular emphasis on the identification of resection cavities and the clinical interpretation of segmentation errors.

A three-dimensional U-Net was developed using four MRI modalities to segment whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC).

The project extends beyond conventional model training by incorporating clinical neuroanatomical review, investigation of false-positive predictions, anatomically motivated post-processing and independent held-out evaluation.

## Principal Findings

The selected model was evaluated on **198 MRI examinations from 92 subjects**, with subject-level separation from the training and validation cohorts.

| Evaluation metric | Result |
|---|---:|
| Baseline macro-average Dice | **0.6525** |
| Baseline resection-cavity Dice | 0.5221 |
| Corrected resection-cavity Dice | **0.5638** |
| Mean paired improvement in RC Dice | **+0.0417** |
| 95% subject-clustered bootstrap confidence interval | **+0.0227 to +0.0625** |
| Reduction in false-positive RC voxels in cavity-absent examinations | **70.2%** |

The post-processing intervention improved average reference-based resection-cavity segmentation performance. However, improvements were not universal: some examinations demonstrated substantial deterioration, and false-positive cavity predictions persisted.

Clinical anatomical review identified errors involving normal cerebrospinal fluid spaces, heterogeneous tumour-associated tissue and a potential disagreement between model predictions and a postoperative reference annotation.

**Central finding:** Improved numerical segmentation performance does not necessarily establish anatomical correctness or clinical reliability. Quantitative evaluation should be complemented by clinically informed examination of model failures.

## Research Documentation

- **[Full Research Report](reports/glioma_segmentation_report.md)** — Experimental methodology, quantitative results, statistical analysis, anatomical interpretation and limitations.
- **[Independent Evaluation Notebook](notebooks/04_model_evaluation.ipynb)** — Held-out evaluation, paired comparisons, subject-clustered bootstrap analysis and clinical failure-case investigation.
- **[Model Development Notebook](notebooks/03_unet_model_development.ipynb)** — Model training, validation experiments and post-processing development.
- **[Evaluation Figures](outputs/figures/)** — Regional performance comparisons, false-positive analysis and anatomical failure-case visualisations.

The model and post-processing intervention are research demonstrations and have not been validated for clinical deployment. Full computational reproduction requires separately obtained imaging data, the trained checkpoint and a compatible software environment.

---

## Clinical Motivation

Gliomas are primary brain tumours with heterogeneous imaging appearances, infiltrative growth patterns and complex treatment-related changes.

Magnetic resonance imaging (MRI) plays a central role in their assessment. Different MRI sequences provide complementary information regarding anatomical structure, contrast enhancement, tissue water content and surrounding signal abnormalities.

Accurate segmentation of tumour-associated abnormalities and postoperative resection cavities can support quantitative assessment of disease extent, longitudinal imaging analysis and future development of computer-assisted neuro-oncological tools.

However, automated segmentation presents challenges beyond identifying abnormal MRI signal.

Postoperative cavities may share imaging characteristics with normal cerebrospinal fluid spaces, while heterogeneous tumour-associated tissue and postoperative enhancement can complicate the distinction between different anatomical regions.

Consequently, a model may achieve reasonable numerical agreement with reference annotations while producing anatomically implausible or clinically important errors.

This project therefore combines deep-learning model development with quantitative evaluation and clinically informed neuroanatomical analysis.

## Project Objectives

The project was designed to investigate the complete workflow of three-dimensional postoperative glioma MRI segmentation, from data preparation to independent evaluation and clinical interpretation.

The principal objectives were to:

1. Explore multimodal brain MRI examinations stored in NIfTI format and examine the complementary information provided by T1-weighted, contrast-enhanced T1-weighted, T2-weighted and T2-FLAIR sequences.
2. Develop a reproducible preprocessing pipeline for multimodal MRI and reference segmentation annotations.
3. Implement and train a three-dimensional U-Net using PyTorch and MONAI.
4. Evaluate segmentation performance across whole tumour, tumour core, enhancing tumour and resection cavity regions.
5. Investigate false-positive predictions and clinically relevant segmentation failures through quantitative analysis and anatomical review.
6. Explore anatomically motivated post-processing strategies to address inconsistent segmentation predictions.
7. Conduct independent held-out evaluation using a predefined protocol, including examination-level comparisons and statistical uncertainty estimation.
8. Identify methodological limitations and future research opportunities in anatomical segmentation, three-dimensional reconstruction and computational neuro-oncology.

The completed study places particular emphasis on the distinction between **segmentation accuracy, anatomical plausibility and clinical validity**.

Although the experimental objectives have been completed, the resulting model remains a research demonstration and is not suitable for clinical decision-making.

---

## Dataset

This project uses the **BraTS 2024 Adult Glioma (BraTS-GLI) training dataset**, obtained through the BraTS 2024 Challenge on Synapse.

The dataset contains multimodal brain MRI examinations and reference segmentation annotations for tumour-associated abnormalities and postoperative resection cavities.

The available dataset comprises **1,350 MRI examinations from 613 subjects**. Some subjects have multiple examinations; therefore, the number of imaging examinations is greater than the number of unique subjects.

### Dataset Partitioning

The dataset was divided into subject-disjoint training, validation and held-out test partitions.

| Partition | MRI examinations | Unique subjects |
|---|---:|---:|
| Training | 936 | 429 |
| Validation | 216 | 92 |
| Held-out test | 198 | 92 |
| **Total** | **1,350** | **613** |

All examinations belonging to a particular subject were assigned to the same partition to prevent subject-level data leakage.

The principal model-development experiment used a subset of **180 training examinations from 180 subjects** and **24 validation examinations from 24 separate subjects**, drawn from their respective partitions.

The held-out test partition was reserved for the final independent evaluation and was not used for model training, checkpoint selection or post-processing development.

### Dataset Availability

The original MRI volumes and segmentation annotations are not included in this GitHub repository. They are stored separately and excluded from Git version control.

**Dataset source:** [BraTS 2024 GLI Training Data — Synapse](https://www.synapse.org/Synapse:syn60086071)

**Dataset version:** V2 (18 June 2024), incorporating segmentation annotation corrections documented in the dataset changelog.

Access to the original dataset is subject to the applicable Synapse and BraTS access conditions.

## MRI Modalities

Each examination includes four complementary MRI sequences used as input channels for the segmentation model.

| Sequence | BraTS identifier | Principal contribution |
|---|---|---|
| T1-weighted | T1n | Anatomical and structural information |
| Contrast-enhanced T1-weighted | T1c | Demonstration of contrast-enhancing tissue |
| T2-weighted | T2w | Sensitive to increased tissue water content |
| T2-FLAIR | T2f | Demonstration of T2-hyperintense abnormalities with suppression of free-fluid signal |

These four modalities are combined into a multichannel volumetric input for the three-dimensional U-Net.

The model uses the complementary imaging information to predict four segmentation regions: whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC).

---

## MRI Preprocessing Pipeline

A reproducible preprocessing workflow was developed using MONAI and PyTorch to prepare multimodal MRI volumes and reference segmentation annotations for three-dimensional deep learning.

The pipeline includes:

- **Image loading:** Reading four MRI modalities and the corresponding reference segmentation from NIfTI files.
- **Spatial standardisation:** Converting imaging volumes to a consistent RAS anatomical orientation.
- **Intensity normalisation:** Normalising nonzero MRI intensities independently for each modality.
- **Foreground cropping:** Removing unnecessary background while retaining the relevant brain anatomy and annotated regions.
- **Training augmentation:** Applying random 96 × 96 × 96 voxel patch sampling, spatial flipping and intensity scaling.
- **Segmentation conversion:** Converting categorical reference annotations into four binary region masks.
- **Data loading:** Preparing multichannel image tensors and segmentation tensors for model training, validation and evaluation.

Random patch sampling and training augmentation were applied during model training. Validation and held-out evaluation instead used full foreground-cropped volumes with sliding-window inference, allowing predictions to be assessed over the complete retained imaging region.

### Segmentation Regions

The supplied categorical segmentation annotations were converted into four binary region masks.

| Region | Description |
|---|---|
| WT | Whole tumour-associated region |
| TC | Tumour core |
| ET | Enhancing tumour |
| RC | Resection cavity |

These regions are represented as four separate output channels.

The segmentation targets are not mutually exclusive. In particular, the tumour-associated regions have nested relationships, while resection-cavity segmentation introduces additional anatomical considerations in postoperative examinations.

Consequently, the model uses a **multilabel segmentation formulation**, with independent sigmoid scores for each region rather than a mutually exclusive softmax classification.

At inference, sigmoid scores are converted into binary predictions using a predefined threshold of 0.5. A separately evaluated post-processing intervention was subsequently applied to the resection-cavity channel to investigate anatomically inconsistent predictions.

### Preprocessing Verification

Initial pipeline verification confirmed the expected tensor dimensions:

| Component | Observed output |
|---|---|
| Training MRI batch | `(2, 4, 96, 96, 96)` |
| Training segmentation batch | `(2, 4, 96, 96, 96)` |
| Representative validation MRI batch | `(1, 4, 133, 185, 132)` |
| Representative validation segmentation batch | `(1, 4, 133, 185, 132)` |

The validation dimensions refer to one representative examination; foreground-cropped dimensions vary between examinations.

Visual inspection demonstrated plausible alignment between MRI anatomy and reference annotations. Automated consistency checks also verified preservation of annotated foreground voxels following segmentation-label conversion.

These checks established the functionality of the preprocessing pipeline but do not constitute comprehensive dataset-wide quality assurance.

**Implementation:** [`02_preprocessing_pipeline.ipynb`](notebooks/02_preprocessing_pipeline.ipynb)

---

## Segmentation Workflow

The completed experimental workflow follows the progression from MRI data preparation through model development, independent evaluation and clinical interpretation.

```text
BraTS 2024 GLI MRI Dataset
             │
             ▼
    Data Exploration
             │
             ▼
    MRI Preprocessing
             │
             ▼
  Four-Channel MRI Input
 (T1n / T1c / T2w / T2f)
             │
             ▼
       3D U-Net
             │
             ▼
   Four Output Logit Maps
    (WT / TC / ET / RC)
             │
             ▼
   Sigmoid + Thresholding
             │
             ▼
    Raw Segmentation Masks
             │
             ├────────────────────┐
             ▼                    ▼
   Baseline Evaluation    Validation-Informed
                          RC Post-Processing
                                  │
                                  ▼
                          Corrected RC Mask
                                  │
             └──────────┬─────────┘
                        ▼
             Independent Held-Out
                Test Evaluation
                        │
                        ▼
              Paired Comparisons
             and Statistical Analysis
                        │
                        ▼
               Anatomical Review
              and Failure Analysis
                        │
                        ▼
             Clinical Interpretation
                and Limitations
```

The trained model generates independent sigmoid scores for four segmentation regions. A predefined threshold of 0.5 produces the baseline binary masks.

A validation-informed relative-confidence correction was developed to address selected inconsistencies between enhancing-tumour and resection-cavity predictions. This intervention modifies only the resection-cavity output and does not require retraining the model.

Both the raw baseline predictions and corrected predictions were evaluated on the same independent held-out test partition. The raw model remained the primary reference analysis, with the post-processing intervention assessed as a secondary comparison.

The evaluation incorporated region-specific segmentation metrics, false-positive analysis, paired examination-level comparisons, subject-clustered bootstrap uncertainty estimation and targeted anatomical review.

## Repository Structure

```text
glioma-mri-segmentation/
├── data/
│   ├── raw/                         # Original dataset excluded from Git
│   ├── interim/                     # Intermediate processing outputs
│   └── processed/                   # Derived data excluded or reviewed
│
├── notebooks/
│   ├── 01_mri_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_unet_model_development.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py             # Reusable MRI preprocessing
│   └── losses.py                    # Multilabel segmentation loss
│
├── models/                          # Model checkpoints excluded from Git
│
├── outputs/
│   └── figures/                     # Evaluation and analysis figures
│
├── reports/
│   └── glioma_segmentation_report.md
│
├── environment.yml                 # Core software environment
├── requirements-lock.txt           # Recorded Python dependency versions
├── README.md
└── .gitignore
```

### Research Documentation

The repository is organised to separate the main stages of the research workflow:

| Component | Purpose |
|---|---|
| `01_mri_data_exploration.ipynb` | Dataset inspection, MRI visualisation and segmentation-label exploration |
| `02_preprocessing_pipeline.ipynb` | Data partitioning, preprocessing, tensor preparation and consistency checks |
| `03_unet_model_development.ipynb` | Model architecture, pilot experiments, training and validation-informed post-processing |
| `04_model_evaluation.ipynb` | Independent held-out evaluation, statistical comparisons and anatomical failure analysis |
| `glioma_segmentation_report.md` | Complete research methodology, results, clinical interpretation and limitations |
| `outputs/figures/` | Figures supporting the quantitative and anatomical analyses |
| [`environment.yml`](environment.yml) | Core Python, PyTorch/CUDA, MONAI and scientific computing dependencies for environment reconstruction |
| [`requirements-lock.txt`](requirements-lock.txt) | Detailed snapshot of the 165 installed Python packages and their recorded versions |

### Data and Checkpoint Availability

The original BraTS MRI dataset, large model checkpoints and local experimental artifacts are stored separately from the public GitHub repository.

The trained checkpoint and full underlying evaluation artifacts are not currently distributed through this repository. Consequently, the published notebooks and documentation support inspection of the methodology, but complete computational reproduction requires access to the original data and the frozen model checkpoint.

The software environment used for this study is documented in [`environment.yml`](environment.yml), with a detailed snapshot of installed Python dependencies in [`requirements-lock.txt`](requirements-lock.txt). The environment specification includes the CUDA 13.0-enabled PyTorch build used for model development and evaluation.

The dependency versions have been checked for consistency, and the YAML configuration has been validated. However, a fresh environment installation and end-to-end computational reproduction have not yet been independently tested. The original MRI dataset, frozen model checkpoint and complete local experimental artifacts remain outside the public repository.

Final repository quality checks, including notebook outputs, relative paths and public accessibility, remain in progress.

---

## Current Progress

The principal experimental study has been completed, including model training, validation-informed post-processing development, independent held-out evaluation and clinical failure analysis.

The research report, evaluation figures and software environment specifications have been completed and committed to the public repository. Remaining work focuses on notebook and file-path auditing, public accessibility checks and final GitHub release preparation.

### Phase 1 — MRI Data Exploration (Completed)

- [x] Establish the project repository and Python environment.
- [x] Configure PyTorch, MONAI, NiBabel and CUDA support.
- [x] Obtain and organise the BraTS 2024 GLI dataset.
- [x] Examine NIfTI image formats and MRI volume dimensions.
- [x] Explore T1n, T1c, T2w and T2f MRI sequences.
- [x] Inspect segmentation annotations and their anatomical relationships.
- [x] Visualise MRI slices and segmentation overlays.

**Notebook:** [`01_mri_data_exploration.ipynb`](notebooks/01_mri_data_exploration.ipynb)

### Phase 2 — MRI Preprocessing (Core Pipeline Completed)

- [x] Establish subject-disjoint training, validation and held-out test partitions.
- [x] Implement anatomical orientation standardisation and intensity normalisation.
- [x] Implement foreground cropping.
- [x] Implement random 3D patch sampling and training augmentation.
- [x] Convert categorical annotations into four binary segmentation regions.
- [x] Verify tensor dimensions and segmentation-label consistency.
- [x] Implement reusable preprocessing components.
- [x] Investigate training-region prevalence and patch sampling.
- [ ] Complete comprehensive dataset-wide quality assurance.

**Notebook:** [`02_preprocessing_pipeline.ipynb`](notebooks/02_preprocessing_pipeline.ipynb)

### Phase 3 — Model Development (Completed)

- [x] Implement a MONAI 3D U-Net architecture.
- [x] Configure multilabel segmentation loss functions.
- [x] Identify and correct the initial categorical-loss mismatch.
- [x] Complete preliminary pilot experiments.
- [x] Implement full-volume sliding-window validation.
- [x] Implement region-specific Dice and false-positive measurements.
- [x] Establish checkpoint selection and experiment logging.
- [x] Complete the 180-training-examination, 24-validation-examination experiment over 40 epochs.
- [x] Select the best checkpoint using validation performance.
- [x] Investigate cavity-aware training and alternative post-processing strategies.
- [x] Finalise the relative-confidence RC correction before held-out testing.
- [x] Verify saved checkpoint integrity and reproducible model predictions.
- [ ] Extract additional experimental components into reusable software modules.

**Notebook:** [`03_unet_model_development.ipynb`](notebooks/03_unet_model_development.ipynb)

### Phase 4 — Independent Model Evaluation (Completed)

- [x] Freeze the selected model checkpoint and evaluation protocol.
- [x] Evaluate 198 held-out MRI examinations from 92 subjects.
- [x] Calculate region-specific Dice scores and false-positive measurements.
- [x] Compare raw and corrected resection-cavity predictions.
- [x] Perform paired examination-level analysis.
- [x] Estimate uncertainty using subject-clustered bootstrap resampling.
- [x] Investigate cavity-present and cavity-absent examinations separately.
- [x] Visualise predictions alongside reference annotations.
- [x] Conduct targeted anatomical review of clinically relevant failures.
- [x] Reproduce and verify a major adverse post-processing case.
- [x] Complete final metric-integrity and evaluation-consistency checks.
- [x] Summarise limitations and clinical implications.

**Notebook:** [`04_model_evaluation.ipynb`](notebooks/04_model_evaluation.ipynb)

### Phase 5 — Research Documentation and GitHub Release (In Progress)

- [x] Prepare the full research report.
- [x] Document the final held-out results and statistical comparisons.
- [x] Document anatomical interpretation, limitations and future directions.
- [x] Generate the final evaluation figures.
- [x] Document the software environment and verify dependency-version consistency.
- [x] Confirm that the intended research report, notebooks and evaluation figures are committed.
- [ ] Finalise the public-facing README.
- [ ] Review notebook outputs, repository paths and public reproducibility.
- [ ] Complete the final GitHub repository quality audit.

**Research report:** [`glioma_segmentation_report.md`](reports/glioma_segmentation_report.md)

**Current milestone:** The experimental research, independent evaluation, research documentation and software environment specification are complete. Remaining work concerns final README refinement, notebook and path auditing, and public GitHub release verification.

---

## Model Development and Experiments

### 3D U-Net Architecture

The segmentation model uses a three-dimensional U-Net implemented with PyTorch and MONAI.

Four MRI modalities form the input channels, while four segmentation regions—whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC)—form the output channels.

Because these regions are not mutually exclusive, the model uses a multilabel formulation. Independent sigmoid functions transform the output logits into region-specific scores, which are converted into binary masks using a predefined threshold.

| Component | Configuration |
|---|---|
| Architecture | MONAI 3D U-Net |
| Trainable parameters | 4,811,129 |
| MRI input | T1n, T1c, T2w, T2f |
| Output regions | WT, TC, ET, RC |
| Loss function | Multilabel Dice + binary cross-entropy |
| Optimiser | AdamW |
| Training patch size | 96 × 96 × 96 voxels |
| Inference | Full-volume sliding-window inference |
| Baseline sigmoid threshold | 0.5 |

### Preliminary Pilot Experiments

Two initial pilot experiments used 24 training examinations and four validation examinations over ten epochs each.

The first experiment used a loss formulation containing categorical cross-entropy, which was subsequently recognised as inappropriate for the overlapping segmentation targets.

A second pilot used multilabel Dice loss combined with binary cross-entropy. This improved preliminary region-specific Dice scores, although segmentation quality remained limited and substantial false-positive predictions persisted.

These pilot experiments established the importance of matching the loss formulation to the anatomical relationships between the segmentation regions. Their small validation cohorts mean that the results should not be interpreted as reliable estimates of generalisable performance.

### Principal Training Experiment

The principal model-development experiment used 180 training examinations from 180 subjects and 24 validation examinations from 24 separate subjects.

| Parameter | Configuration |
|---|---|
| Training subset | 180 examinations / 180 subjects |
| Validation subset | 24 examinations / 24 subjects |
| Training duration | 40 epochs |
| Initial learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Loss | Multilabel Dice + binary cross-entropy |
| Validation frequency | Every epoch |
| Checkpoint selection | Macro-average of present-region mean Dice |
| Selected checkpoint | Epoch 38 |
| Sliding-window patch size | 96 × 96 × 96 voxels |
| Sliding-window overlap | 0.25 |
| Baseline probability threshold | 0.5 |

The selected model checkpoint was frozen before independent held-out evaluation.

The principal experiment was intended to establish a complete and evaluable segmentation workflow rather than demonstrate optimal performance across alternative architectures or training-set sizes.

### Validation-Informed Post-Processing

An important focus of the study was the difficulty of distinguishing postoperative resection cavities from enhancing tumour-associated tissue and other structures with overlapping imaging characteristics.

The model produces independent sigmoid scores for enhancing tumour and resection cavity. Consequently, the raw binary predictions may assign both labels to the same voxel.

Several development-stage strategies were investigated, including cavity-aware training and alternative post-processing rules.

The final intervention was a **relative-confidence resection-cavity correction**, selected using validation information before the held-out test evaluation.

The rule retains a predicted RC voxel when either:

- the voxel is not simultaneously predicted as ET; or
- the RC sigmoid score is at least as high as the ET sigmoid score.

Where ET and RC overlap and the ET score is higher, the voxel is removed from the RC prediction.

Expressed mathematically:

\[
RC_{\mathrm{corrected}}
=
RC_{\mathrm{raw}}
\land
\left(
\neg ET_{\mathrm{raw}}
\lor
P(RC)\geq P(ET)
\right)
\]

This correction modifies only the RC segmentation mask. The U-Net weights and the WT, TC and ET predictions remain unchanged.

The sigmoid scores are treated as relative model-confidence measures, not necessarily calibrated clinical probabilities.

### Independent Held-Out Evaluation

The frozen model and predefined post-processing rule were evaluated on 198 MRI examinations from 92 subjects who were not included in the training or validation partitions.

The raw model predictions formed the primary analysis. Corrected predictions were evaluated as a secondary, paired comparison on the same examinations.

| Evaluation outcome | Raw model | Corrected model |
|---|---:|---:|
| Macro-average Dice | 0.6525 | 0.6630 |
| RC Dice in RC-present examinations | 0.5221 | 0.5638 |
| RC precision | 0.4872 | 0.5672 |
| RC recall | 0.6652 | 0.6470 |

Among the 157 examinations containing a reference resection cavity, the mean paired change in RC Dice was **+0.0417**, with a subject-clustered bootstrap 95% confidence interval of **+0.0227 to +0.0625**.

At examination level:

- 111 examinations improved.
- 22 examinations worsened.
- 24 examinations were unchanged.

Among the 41 examinations without a reference resection cavity, the correction reduced the total number of false-positive RC voxels from 496,653 to 147,885—a reduction of approximately **70.2%**.

However, all 41 cavity-absent examinations retained some false-positive RC predictions.

These findings indicate improved average reference-based RC segmentation performance, accompanied by a precision–recall trade-off and persistent false-positive errors.

### Clinical Failure Analysis

Aggregate segmentation metrics concealed important variation between individual examinations.

One examination, `BraTS-GLI-02119-101`, demonstrated a particularly substantial deterioration following post-processing, with RC Dice decreasing from approximately 0.5345 to 0.0046.

Targeted re-inference reproduced the saved quantitative results.

Anatomical review identified enhancing tissue that was discordant with the reference RC annotation, raising the possibility of annotation ambiguity. This observation does not establish that the reference annotation was incorrect.

The case demonstrates that a rule which improves average performance can nevertheless produce clinically important individual failures.

The study therefore emphasises that segmentation metrics should be interpreted alongside anatomical plausibility, error distribution and the limitations of reference annotations.

### Experimental Interpretation

The completed experiments support three principal conclusions:

1. A reproducible postoperative glioma segmentation workflow can be developed using multimodal MRI and a 3D U-Net.
2. Validation-informed relative-confidence post-processing can improve average reference-based resection-cavity segmentation without retraining the model.
3. Improvements in aggregate Dice scores do not guarantee anatomical correctness or consistent performance across individual examinations.

The findings are exploratory and specific to the present dataset, model and evaluation protocol. They do not establish clinical safety, external generalisability or suitability for clinical deployment.

For complete methodology, validation experiments, statistical analysis and clinical interpretation, see the [full research report](reports/glioma_segmentation_report.md).

---

## Future Directions

Following completion of the core segmentation experiments and independent held-out evaluation, this project provides a technical foundation for further investigations in computational neuro-oncology.

The proposed extensions focus on translating volumetric MRI segmentation into three-dimensional anatomical representations, investigating spatial patterns of glioma recurrence and improving the robustness and clinical interpretability of automated imaging analysis.

These are future research directions rather than completed components of the present study. Each would require additional development and independent validation.

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

- exploration of physical models for anatomical visualization, education, and communication.

This extension remains separate from the completed core segmentation study. Although the segmentation model has undergone independent held-out evaluation, the observed anatomical errors and substantial individual-case variation must be considered before using its predictions for three-dimensional reconstruction.

Future work should investigate how segmentation uncertainty, false-positive regions and anatomical inaccuracies propagate into reconstructed meshes and physical models. Any potential clinical application would require additional validation of the resulting anatomical representations.

### 2. Spatial Prediction of Glioma Recurrence

A longer-term research direction is to extend tumour segmentation towards prediction of the spatial patterns of glioma recurrence and progression.

Glioblastoma is an infiltrative disease in which tumour cells may extend beyond abnormalities visible on conventional MRI. Of particular interest is whether

information from the visible tumour, surrounding tissue, and cerebral white-matter architecture can be combined to identify preferential pathways of subsequent progression.

Recent research has investigated ****diffusion tensor imaging (DTI)****, tractography, and computational modelling of tumour–white-matter relationships as potential approaches for studying these pathways. Machine-learning and deep-learning methods provide additional means of extracting and integrating multimodal spatial features.

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

A particularly interesting extension would be to investigate whether detailed ****three-dimensional tumour segmentation**** provides useful spatial information

for recurrence modelling beyond tumour localization alone.

This represents a progression from asking:

> ****Where is the tumour now?****

to investigating:

> ****Where is the tumour likely to progress or recur?****

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

The project was developed using the following technologies:

| Technology | Role |
|---|---|
| Python | Primary programming language |
| PyTorch | Deep-learning model implementation and training |
| MONAI | Medical image preprocessing, augmentation, model architecture and sliding-window inference |
| NiBabel | Reading and inspecting NIfTI medical imaging files |
| NumPy | Numerical operations and array processing |
| Matplotlib | MRI visualisation and research figures |
| Pandas | Evaluation results, tabulation and statistical analysis |
| Jupyter Notebook | Interactive experimentation, model development and evaluation |
| CUDA | GPU-accelerated model training and inference |
| Git and GitHub | Version control, research documentation and portfolio presentation |

Model development and evaluation were performed using a local NVIDIA GPU-enabled workstation.

The principal software dependencies are documented in [`environment.yml`](environment.yml), including the CUDA-enabled PyTorch build used for model development and evaluation. A detailed snapshot of installed Python packages is provided in [`requirements-lock.txt`](requirements-lock.txt).

The environment specification has undergone syntax and dependency-version consistency checks, but recreation in a fresh environment has not yet been independently tested. The original imaging dataset and trained model checkpoint are stored separately from the public repository, limiting complete computational reproducibility.

---

## Project Philosophy

The objective of this project extends beyond training a deep-learning model to generate MRI segmentations.

The study was designed to demonstrate an understanding of the complete medical-imaging machine-learning workflow: dataset exploration, preprocessing, model development, experimental design, independent evaluation, statistical analysis and critical clinical interpretation.

Particular emphasis was placed on three principles:

1. **Scientific reproducibility:** Documenting experimental decisions, preserving evaluation protocols and distinguishing development data from independent test data.
2. **Clinical interpretability:** Examining segmentation predictions in their anatomical context rather than relying exclusively on aggregate performance metrics.
3. **Critical evaluation:** Reporting adverse outcomes, false-positive predictions and methodological limitations alongside improvements in quantitative performance.

The resection-cavity post-processing experiments illustrate these principles. Although the selected correction improved average Dice scores, it also produced substantial deterioration in some individual examinations.

This distinction between numerical performance and anatomical reliability is central to the study.

The longer-term aim is to investigate how computational imaging methods can contribute to neuro-oncological research, three-dimensional anatomical reconstruction, longitudinal imaging analysis and other clinically meaningful applications.

The project is presented as an exploratory research contribution and a foundation for further investigation, rather than a clinically validated segmentation system.

---

## Project Status

**Completed experimental study — final repository preparation in progress.**

The principal research milestones have been completed:

- MRI dataset exploration and preprocessing pipeline development.
- Implementation and training of a multilabel 3D U-Net.
- Completion of the 40-epoch principal training experiment.
- Validation-informed development of resection-cavity post-processing.
- Independent evaluation on 198 MRI examinations from 92 subjects.
- Statistical analysis, false-positive assessment and anatomical failure-case review.
- Preparation of the full research report and evaluation figures.
- Documentation and version-consistency verification of the software environment.

Remaining repository tasks include finalising the README, reviewing notebook outputs and file paths, verifying public accessibility of the research artifacts and completing the final GitHub quality audit. Independent recreation of the software environment and complete computational reproduction remain unverified.

The Git commit history documents the project's progression from initial MRI exploration through model development, independent evaluation and clinical interpretation.

**Clinical-use disclaimer:** This project is intended for research and educational purposes. Its predictions have not been validated for clinical diagnosis, treatment planning or patient management.

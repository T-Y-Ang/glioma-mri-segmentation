# Glioma MRI Segmentation — Research Report

**Project:** Deep Learning for Glioma MRI Segmentation  
**Dataset:** BraTS 2024 Glioma (GLI)  
**Framework:** PyTorch and MONAI  
**Model:** 3D U-Net  
**Status:** Baseline training completed; preliminary validation and clinical error analysis completed; held-out testing pending  
**Last updated:** 16 October 2026

> **Research status:** This is a living research report documenting the methodology, experimental decisions, quantitative results, clinical observations, limitations and future directions of the project. Current results are derived from a limited validation subset and do not establish clinical readiness.

---

## 1. Project Overview and Clinical Motivation

### 1.1 Background

Glioma segmentation is an important component of neuro-oncological imaging research. Delineation of tumour-associated abnormalities and postoperative resection cavities may support quantitative assessment of disease extent, longitudinal imaging analysis and future development of computer-assisted surgical planning tools.

Manual segmentation requires considerable expertise and time. Deep learning provides a potential means of automating segmentation, but numerical agreement with reference annotations does not necessarily guarantee anatomical plausibility or clinical usefulness.

This project was therefore designed not only to develop a segmentation model, but also to evaluate its predictions through a combination of quantitative metrics and clinical neuroanatomical interpretation.

### 1.2 Objectives

The principal objectives are:

1. Establish a reproducible pipeline for multimodal brain MRI preprocessing and segmentation.
2. Develop and train a 3D U-Net using the BraTS 2024 Glioma dataset.
3. Evaluate segmentation performance across tumour-associated regions and resection cavities.
4. Investigate segmentation errors through anatomical review and quantitative analysis.
5. Explore improvements to segmentation consistency and clinically relevant false-positive predictions.
6. Investigate future applications in 3D visualisation, surgical planning and longitudinal imaging research.

### 1.3 Project philosophy

The project emphasises the distinction between **numerical model performance** and **clinically meaningful model behaviour**.

A model may achieve a reasonable Dice score while producing anatomically implausible predictions. Conversely, apparently small numerical errors may be clinically important if they involve critical anatomical structures.

For this reason, quantitative evaluation is complemented by examination of individual MRI studies and predicted segmentation masks.

---

## 2. Dataset and Data Management

### 2.1 Dataset source

The project uses the **BraTS 2024 Glioma (GLI)** training dataset obtained through Synapse.

The dataset provides multimodal MRI examinations with reference segmentation annotations.

Four MRI modalities are used:

| Modality | Description |
|---|---|
| T1n | Native T1-weighted MRI |
| T1c | Contrast-enhanced T1-weighted MRI |
| T2w | T2-weighted MRI |
| T2f | T2-FLAIR MRI |

These modalities provide complementary information regarding anatomical structure, enhancement, tumour-associated signal abnormalities and fluid characteristics.

### 2.2 Dataset inventory

The dataset inventory identified:

| Characteristic | Count |
|---|---:|
| MRI examinations | 1,350 |
| Unique subjects | 613 |

Some subjects contribute multiple examinations.

This creates a risk of data leakage if examinations from the same subject are distributed across training and evaluation cohorts.

### 2.3 Subject-level dataset splitting

To minimise this risk, the dataset was divided using subject identifiers rather than examination-level randomisation.

A fixed random seed of 42 was used.

| Cohort | Examinations | Unique subjects |
|---|---:|---:|
| Training | 936 | 429 |
| Validation | 216 | 92 |
| Held-out test | 198 | 92 |
| **Total** | **1,350** | **613** |

The resulting cohorts were checked for subject overlap.

No subject overlap was identified between the training, validation and test cohorts.

The split manifest was saved locally in:

`data/processed/subject_level_splits.json`

This manifest includes local imaging paths and should not be published without removing or replacing environment-specific information.

### 2.4 Initial experimental subset

For the first training experiment, a smaller subset was selected to establish the full modelling and evaluation workflow.

| Cohort | Examinations |
|---|---:|
| Training subset | 180 |
| Validation subset | 24 |

The validation subset contained:

| Region | Reference present | Reference absent |
|---|---:|---:|
| Whole tumour | 24 | 0 |
| Tumour core | 19 | 5 |
| Enhancing tumour | 19 | 5 |
| Resection cavity | 19 | 5 |

This initial experiment was intended to establish a reproducible baseline, identify implementation problems and investigate early model behaviour.

The held-out test cohort has not yet been evaluated.

---

## 3. Segmentation Regions and Label Representation

### 3.1 Original reference labels

The original segmentation labels are:

| Label value | Tissue category |
|---|---|
| 0 | Background |
| 1 | Non-enhancing tumour core |
| 2 | FLAIR hyperintensity |
| 3 | Enhancing tumour |
| 4 | Resection cavity |

### 3.2 Model output regions

The reference segmentation is converted into four binary masks:

| Region | Abbreviation | Original labels included |
|---|---|---|
| Whole tumour | WT | 1, 2, 3 |
| Tumour core | TC | 1, 3 |
| Enhancing tumour | ET | 3 |
| Resection cavity | RC | 4 |

The three tumour-associated regions are nested:

**ET ⊆ TC ⊆ WT**

Resection cavity is represented separately.

### 3.3 Implications for model architecture

The model predicts four binary masks using independent sigmoid output channels.

This representation allows overlapping predictions.

Overlap is expected between WT, TC and ET because these regions are hierarchically related.

However, the same output representation also permits simultaneous ET and RC predictions at individual voxels.

In the reference segmentation, ET and RC are mutually exclusive.

The absence of an explicit ET–RC consistency constraint became an important finding during subsequent error analysis.

---

## 4. Preprocessing and Data Augmentation

### 4.1 Implementation

Preprocessing was implemented using MONAI.

Reusable transformations are maintained in:

`src/preprocessing.py`

### 4.2 Validation preprocessing

The deterministic validation pipeline includes:

- Loading the four MRI modalities and reference segmentation.
- Standardising image orientation.
- Cropping foreground regions.
- Normalising MRI intensities.
- Converting original segmentation labels into four binary output regions.

Using deterministic validation preprocessing supports reproducible comparisons between checkpoints and post-processing interventions.

### 4.3 Training preprocessing

The training pipeline uses corresponding preprocessing operations with additional augmentation.

Training patches have dimensions of:

**96 × 96 × 96 voxels**

Positive and negative patch sampling is used, with two patches generated per examination.

Additional augmentation includes random spatial flipping and intensity scaling.

### 4.4 Rationale for patch-based training

Three-dimensional MRI volumes can be computationally demanding.

Patch-based training reduces memory requirements and allows volumetric segmentation models to be trained using a consumer GPU.

However, restricting the model's spatial field of view may reduce its ability to distinguish structures with similar local MRI appearances but different anatomical contexts.

This became a possible explanatory hypothesis during the investigation of normal CSF spaces being misclassified as resection cavities.

The hypothesis has not yet been tested experimentally.

---

## 5. Model Architecture and Training Configuration

### 5.1 Baseline model

A 3D U-Net was implemented using MONAI.

| Parameter | Configuration |
|---|---|
| Architecture | 3D U-Net |
| Input channels | 4 |
| Output channels | 4 |
| Encoder channels | 16, 32, 64, 128, 256 |
| Strides | 2, 2, 2, 2 |
| Residual units per stage | 2 |
| Trainable parameters | 4,811,129 |
| Output activation | Sigmoid |

The four input channels correspond to the MRI modalities.

The four output channels correspond to WT, TC, ET and RC.

### 5.2 Loss function

The training objective combines Dice loss and binary cross-entropy loss.

The implementation is maintained in:

`src/losses.py`

The loss weights are:

| Component | Weight |
|---|---:|
| Dice loss | 1.0 |
| Binary cross-entropy loss | 1.0 |

The combined loss was intended to encourage segmentation overlap while also providing voxel-level classification supervision.

### 5.3 Optimisation

The model was trained using AdamW.

| Hyperparameter | Value |
|---|---:|
| Learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Training epochs | 40 |

### 5.4 Computing environment

Training was conducted locally using:

- NVIDIA RTX 4080 SUPER GPU with 16 GB VRAM.
- 32 GB system RAM.
- Windows Subsystem for Linux running Ubuntu.
- Python 3.11.
- PyTorch and MONAI.

The project demonstrates a workflow designed to operate on consumer workstation hardware.

---

## 6. Baseline Training Experiment

### 6.1 Experiment identification

The completed experiment is identified as:

`unet_180train_24val_20261013_203139`

The experiment records include model checkpoints, training history, region-level validation results and configuration information.

Large model checkpoints and raw MRI datasets are maintained outside the public GitHub repository.

### 6.2 Training progress

The baseline model completed 40 training epochs.

Training loss decreased from approximately 1.68 during the initial epoch to 0.5275 at epoch 40.

Validation macro Dice improved substantially over the course of training, with fluctuations during later epochs.

The best validation checkpoint was obtained at epoch 38.

### 6.3 Training results

| Measurement | Result |
|---|---:|
| Best checkpoint epoch | 38 |
| Best validation macro Dice | 0.680868 |
| Final epoch | 40 |
| Final validation macro Dice | 0.661856 |
| Final training loss | 0.527487 |

The best checkpoint and final checkpoint were saved separately.

Checkpoint loading and prediction consistency were verified, allowing subsequent analysis without repeating model training.

### 6.4 Interpretation

The training results demonstrate that the model learned meaningful segmentation features.

However, the relatively small validation subset limits the conclusions that can be drawn about generalisation.

The observed training and validation curves do not establish that the model is adequately optimised or free from overfitting.

More comprehensive validation is required.

---

## 7. Quantitative Validation

### 7.1 Evaluation method

Full-volume inference was performed using MONAI sliding-window inference.

| Parameter | Value |
|---|---|
| ROI size | 96 × 96 × 96 |
| Sliding-window overlap | 0.25 |
| Sliding-window batch size | 1 |
| Sigmoid threshold | 0.5 |

The epoch-38 checkpoint was used for detailed validation and clinical error analysis.

### 7.2 Reproducibility

The saved checkpoint-selection score was reproduced in a separate validation pass.

| Measurement | Value |
|---|---:|
| Saved macro Dice | 0.680868 |
| Reproduced macro Dice | 0.680869 |
| Absolute difference | Approximately 0.00000038 |

This supports consistency between the saved checkpoint, preprocessing and evaluation pipeline.

### 7.3 Region-specific performance

| Region | Mean Dice | Mean precision | Mean recall |
|---|---:|---:|---:|
| WT | 0.8546 | 0.8705 | 0.8488 |
| TC | 0.6548 | 0.6956 | 0.6713 |
| ET | 0.6736 | 0.6920 | 0.6977 |
| RC | 0.5405 | 0.5377 | 0.6490 |

These means were calculated among examinations containing the corresponding reference region.

Whole-tumour segmentation achieved the highest Dice.

Resection-cavity segmentation demonstrated the weakest Dice and precision.

This motivated a more detailed investigation of false-positive RC predictions.

### 7.4 False positives in region-absent examinations

| Region | Absent examinations | Cases with false positives | Total false-positive voxels |
|---|---:|---:|---:|
| TC | 5 | 3 | 1,869 |
| ET | 5 | 3 | 1,635 |
| RC | 5 | 5 | 102,424 |

The RC false-positive burden was substantially larger than the corresponding TC and ET false-positive burdens.

The original training history recorded 102,425 RC false-positive voxels at epoch 38. A separate validation pass produced 102,424.

A one-voxel difference was also observed between separate inference runs on an individual examination.

The exact cause of these minor discrepancies has not been established.

### 7.5 Limitation of the checkpoint-selection metric

The checkpoint-selection score averages Dice across regions using only examinations where the reference region is present.

Consequently, false-positive predictions in examinations where a region is absent do not directly reduce that region's contribution to the checkpoint-selection metric.

This explains how a model can achieve a reasonable aggregate Dice while generating substantial false-positive predictions in region-absent examinations.

Additional measures are therefore needed to characterise clinically meaningful errors.

---

## 8. Clinical Error Analysis

### 8.1 Rationale

Aggregate segmentation metrics quantify disagreement between predicted and reference masks but do not identify the anatomical structures responsible.

Individual validation examinations were therefore reviewed using MRI sequences and predicted segmentation overlays.

This review incorporated clinical neuroanatomical interpretation to identify the anatomical distribution of false-positive predictions.

Two principal patterns emerged:

1. Normal CSF-space-associated misclassification.
2. Tumour-associated misclassification.

These patterns may coexist within the same examination.

### 8.2 Distribution of false-positive RC predictions

Five validation examinations contained no annotated resection cavity.

Their false-positive RC predictions were distributed as follows:

| Examination | False-positive RC voxels | Contribution |
|---|---:|---:|
| BraTS-GLI-02116-101 | 67,825 | 66.22% |
| BraTS-GLI-00469-100 | 21,765 | 21.25% |
| BraTS-GLI-02609-101 | 11,892 | 11.61% |
| BraTS-GLI-02736-101 | 637 | 0.62% |
| BraTS-GLI-02217-103 | 305 | 0.30% |

The two largest examinations accounted for approximately 87.47% of the false-positive RC volume.

The three largest examinations accounted for approximately 99.08%.

### 8.3 Error Pattern A — Normal CSF Spaces Misclassified as Resection Cavities

#### 8.3.1 Sylvian fissure misclassification

In examination `BraTS-GLI-02736-101`, the model predicted 637 RC voxels despite the absence of an annotated resection cavity.

Review of the contrast-enhanced T1-weighted MRI with the predicted RC overlay identified false-positive clusters corresponding to the **Sylvian fissure**, a normal CSF-containing anatomical structure.

This error was recognised through clinical neuroanatomical interpretation of the imaging.

Connected-component analysis identified eight predicted RC components.

The two largest contained:

- 343 voxels.
- 239 voxels.

Together, these accounted for approximately **91.4%** of the false-positive RC volume.

The spatial organisation of these predictions suggests that the error was not simply isolated voxel noise.

**Clinical interpretation:** The model appears to have misclassified a recognisable normal CSF-containing structure as a postoperative cavity.

This illustrates the importance of anatomical review beyond numerical metrics.

#### 8.3.2 Ventricular-associated misclassification

In examination `BraTS-GLI-02609-101`, the model predicted approximately 11,893 RC voxels during a separate inference run, despite the absence of an annotated cavity.

Review of axial T1-contrast and FLAIR imaging demonstrated a prominent false-positive region appearing to involve the lateral ventricular CSF space, with another focus near the interhemispheric fissure.

A mutually exclusive anatomical breakdown showed:

| Reference region | Percentage of predicted RC |
|---|---:|
| Enhancing tumour | 30.08% |
| TC excluding ET | 0.20% |
| WT excluding TC | 0.78% |
| Outside WT | 68.94% |

The imaging appearance supports a ventricular CSF-space-associated error.

However, the quantitative finding that 68.94% of predicted RC lies outside WT does not prove that all those voxels correspond to CSF.

This examination may contain a mixture of CSF-space-associated and tumour-associated false positives.

#### 8.3.3 Possible explanation

Normal CSF spaces and postoperative resection cavities can share certain fluid-related MRI characteristics.

A segmentation model may therefore have difficulty distinguishing them if it relies heavily on local image intensity and texture.

Limited spatial context during patch-based training is another possible contributing factor.

These explanations remain hypotheses rather than demonstrated mechanisms.

### 8.4 Error Pattern B — Tumour Tissue Misclassified as Resection Cavity

The two examinations with the greatest false-positive RC volumes demonstrated a different anatomical pattern.

In both, predicted RC masks overlapped extensive heterogeneous tumour-associated abnormalities.

#### 8.4.1 Examination BraTS-GLI-02116-101

The model predicted 67,825 RC voxels without an annotated cavity.

MRI review demonstrated a large heterogeneous lesion with irregular enhancement and central low-signal regions.

Quantitative overlap with the reference segmentation was:

| Reference region | Percentage of predicted RC |
|---|---:|
| WT | 98.55% |
| TC | 98.07% |
| ET | 91.88% |
| Outside WT | 1.45% |

WT, TC and ET are nested, so these percentages are not additive.

This examination alone accounted for 66.22% of the total RC false-positive volume among RC-absent examinations.

#### 8.4.2 Examination BraTS-GLI-00469-100

The model predicted 21,765 RC voxels without an annotated cavity.

MRI review demonstrated an irregularly enhancing lesion with heterogeneous internal signal.

A mutually exclusive breakdown showed:

| Reference region | Predicted RC voxels | Percentage |
|---|---:|---:|
| ET | 18,350 | 84.31% |
| TC excluding ET | 3,352 | 15.40% |
| WT excluding TC | 32 | 0.15% |
| Outside WT | 31 | 0.14% |

Thus, **99.86%** of the false-positive RC predictions overlapped annotated tumour tissue.

#### 8.4.3 Possible explanation

The model may struggle to distinguish postoperative cavities from heterogeneous tumour-associated tissue.

Intratumoural necrosis or other tissue heterogeneity may contribute to the imaging similarities.

However, the exact mechanism cannot be established from the available overlap statistics and selected MRI slices.

### 8.5 Simultaneous ET–RC Predictions

The model uses independent sigmoid outputs for ET and RC.

A dedicated analysis across all 24 validation examinations identified **20 examinations containing at least some simultaneous ET–RC predictions**.

Among RC-absent examinations:

| Examination | Predicted RC also labelled ET | Predicted RC overlapping reference ET |
|---|---:|---:|
| 02116-101 | 96.90% | 91.88% |
| 00469-100 | 88.42% | 84.31% |
| 02609-101 | 30.96% | 30.08% |
| 02736-101 | 0.00% | 0.00% |
| 02217-103 | 0.00% | 0.00% |

These columns describe different measurements.

**Predicted ET–RC overlap** measures voxels simultaneously assigned both labels by the model.

**Predicted RC overlapping reference ET** measures how much predicted RC lies within annotated enhancing tumour.

The two largest tumour-associated false-positive examinations showed particularly extensive simultaneous ET–RC predictions.

### 8.6 ET–RC overlap by reference cavity presence

The validation cohort was divided according to whether an annotated resection cavity was present.

| Measurement | RC absent | RC present |
|---|---:|---:|
| Examinations | 5 | 19 |
| Cases with simultaneous ET–RC predictions | 3 | 17 |
| Total predicted RC voxels | 102,425 | 486,401 |
| Simultaneous ET–RC voxels | 88,647 | 56,393 |
| Voxel-weighted ET–RC overlap | 86.55% | 11.59% |

The voxel-weighted ET–RC overlap percentage was approximately **7.5 times higher** in RC-absent examinations.

This suggests an association between overlapping predictions and false-positive cavity segmentation.

However, the association does not establish causality.

The finding motivated a controlled post-processing experiment using the existing model checkpoint.

---

## 9. Anatomical-Consistency Post-processing Experiment

### 9.1 Rationale

The clinical error analysis identified frequent simultaneous ET and RC predictions.

These predictions were especially prominent in examinations without annotated resection cavities.

A simple post-processing intervention was therefore investigated to determine whether enforcing ET–RC exclusivity could improve segmentation performance.

### 9.2 Method

The existing epoch-38 model checkpoint was used without retraining.

The proposed rule removes predicted RC voxels wherever the model simultaneously predicts ET:

`RC_corrected = RC_predicted AND NOT ET_predicted`

The original and corrected predictions were evaluated using the same 24 validation examinations.

The correction modifies only the predicted RC mask.

### 9.3 False-positive reduction in RC-absent examinations

In the five RC-absent examinations:

| Measurement | Original | After correction |
|---|---:|---:|
| Predicted RC voxels | 102,425 | 13,778 |
| False-positive volume reduction | — | 86.55% |

Case-level results were:

| Examination | Original RC voxels | Remaining RC voxels | Reduction |
|---|---:|---:|---:|
| 02116-101 | 67,825 | 2,105 | 96.90% |
| 00469-100 | 21,765 | 2,520 | 88.42% |
| 02609-101 | 11,893 | 8,211 | 30.96% |
| 02736-101 | 637 | 637 | 0.00% |
| 02217-103 | 305 | 305 | 0.00% |

The correction substantially reduced false-positive volume in the two largest tumour-associated examinations.

However, it did not affect the Sylvian fissure-associated error.

The ventricular-associated examination retained a substantial false-positive volume.

This reinforces the interpretation that different anatomical error patterns may require different corrective approaches.

### 9.4 Performance in RC-present examinations

The correction was evaluated in 19 examinations containing annotated resection cavities.

| Metric | Original | Corrected | Change |
|---|---:|---:|---:|
| Mean Dice | 0.5405 | 0.5748 | +0.0343 |
| Mean precision | 0.5377 | 0.5905 | +0.0528 |
| Mean recall | 0.6490 | 0.6261 | -0.0229 |

Mean Dice increased by approximately 6.3% relative to the original value.

Precision also improved, while recall decreased.

This demonstrates a trade-off between false-positive suppression and preservation of genuine cavity predictions.

### 9.5 True-positive and false-positive voxels removed

Across the 19 RC-present examinations, the correction removed:

| Removed voxel category | Count |
|---|---:|
| False-positive RC voxels | 53,820 |
| True-positive RC voxels | 2,573 |
| **Total removed** | **56,393** |

Approximately 95.4% of removed voxels were false positives.

Approximately 4.6% were true positives.

The high proportion of false-positive removal explains the overall improvement in precision and Dice.

However, the removal of correctly segmented cavity tissue demonstrates that the rule is not risk-free.

### 9.6 Case-level outcomes

Among the 19 RC-present examinations:

| Outcome | Number of examinations |
|---|---:|
| Dice improved | 13 |
| Dice deteriorated | 4 |
| Dice unchanged | 2 |

The correction therefore improved Dice in most, but not all, RC-present examinations.

#### Largest improvements

| Examination | Original Dice | Corrected Dice | Change |
|---|---:|---:|---:|
| 02382-100 | 0.2360 | 0.5014 | +0.2654 |
| 02394-100 | 0.6632 | 0.7741 | +0.1109 |
| 02375-101 | 0.3490 | 0.4452 | +0.0962 |
| 02553-101 | 0.7782 | 0.8524 | +0.0742 |
| 02105-104 | 0.8164 | 0.8710 | +0.0546 |

#### Cases with deteriorating Dice

| Examination | Original Dice | Corrected Dice | Dice change | Recall change |
|---|---:|---:|---:|---:|
| 02817-100 | 0.0633 | 0.0481 | -0.0152 | -0.1845 |
| 02158-101 | 0.0151 | 0.0004 | -0.0147 | -0.0076 |
| 02260-100 | 0.1355 | 0.1300 | -0.0055 | -0.0160 |
| 02726-100 | 0.5861 | 0.5857 | -0.0004 | -0.0041 |

The largest absolute recall reduction occurred in examination `BraTS-GLI-02817-100`.

Its reference cavity contained only 1,046 voxels, and recall decreased by 18.45 percentage points.

In `BraTS-GLI-02158-101`, Dice decreased from 0.0151 to 0.0004, indicating that the already poor segmentation lost nearly all remaining overlap with the reference cavity.

These examples demonstrate why improvement in mean Dice alone is insufficient to establish the clinical suitability of a correction.

### 9.7 Effect of reference cavity size

An exploratory analysis examined whether reference cavity size was associated with changes in segmentation performance.

The 19 RC-present examinations were divided into three groups based on reference cavity voxel count.

These groups represent relative categories within this validation subset, not clinically validated size thresholds.

| Size group | Examinations | Median reference RC voxels | Mean Dice change | Mean recall change | Cases with worsened Dice |
|---|---:|---:|---:|---:|---:|
| Small | 7 | 2,384 | +0.0645 | -0.0520 | 1 |
| Medium | 6 | 16,225 | +0.0172 | -0.0084 | 3 |
| Large | 6 | 42,671 | +0.0161 | -0.0034 | 0 |

Spearman rank correlations were:

| Relationship | Spearman correlation |
|---|---:|
| Reference RC size versus Dice change | 0.0307 |
| Reference RC size versus recall change | 0.1186 |

These results did not demonstrate a meaningful monotonic relationship between reference cavity size and Dice improvement.

The small-cavity group had the greatest mean Dice improvement but also the greatest mean recall reduction.

This may reflect the simultaneous removal of substantial false-positive volume and some correctly identified cavity tissue.

The small sample size limits interpretation.

### 9.8 Interpretation

The ET–RC exclusion experiment produced three principal findings:

1. **False-positive RC volume was substantially reduced in RC-absent examinations**, particularly those demonstrating tumour-associated misclassification.
2. **Mean Dice and precision improved in RC-present examinations**, demonstrating that the correction can improve quantitative segmentation performance.
3. **Genuine cavity voxels were also removed**, causing reduced recall and deterioration in some examinations.

These findings support further investigation of anatomically consistent segmentation methods.

However, they do not justify unconditional adoption of the post-processing rule.

The intervention was developed and evaluated using the same limited validation subset.

Its performance on independent data remains unknown.

---

## 10. Integrated Clinical Interpretation

### 10.1 Distinct error patterns require distinct solutions

The evaluation identified at least two anatomically different forms of RC misclassification.

**CSF-space-associated errors** involved normal anatomical structures, including the Sylvian fissure and probable ventricular CSF spaces.

**Tumour-associated errors** involved heterogeneous tumour abnormalities, frequently accompanied by simultaneous ET–RC predictions.

The ET–RC exclusion experiment was effective against much of the tumour-associated false-positive burden but did not address the Sylvian fissure-associated error.

This suggests that a single correction may be insufficient to address all clinically meaningful segmentation errors.

### 10.2 Anatomical consistency and segmentation accuracy

The independent sigmoid output representation allows the model to generate anatomically incompatible predictions.

Enforcing ET–RC exclusivity through post-processing improved average quantitative performance.

However, the intervention also removed correctly segmented cavity tissue.

This illustrates an important distinction between improving anatomical consistency and improving segmentation accuracy.

An anatomically motivated rule may reduce certain errors while introducing others.

### 10.3 Contribution of clinical expertise

The initial quantitative analysis identified RC as the weakest-performing segmentation region.

Direct MRI review subsequently identified specific anatomical structures and tumour-associated abnormalities responsible for some false-positive predictions.

In particular, clinical recognition of the Sylvian fissure as a normal CSF-containing structure revealed an error that would not have been adequately characterised by aggregate Dice alone.

This observation helped distinguish CSF-space-associated misclassification from tumour-associated errors and guided the subsequent investigation of ET–RC prediction overlap.

The project therefore illustrates how clinical neuroanatomical knowledge can contribute to model evaluation, hypothesis generation and experimental design.

---

## 11. Limitations

### 11.1 Limited validation sample

The current evaluation uses only 24 validation examinations.

This subset may not represent the full diversity of tumour morphology, postoperative appearances or imaging characteristics in the dataset.

### 11.2 Held-out test cohort not yet evaluated

The independent test cohort remains untouched.

The current findings should not be interpreted as final generalisation performance.

### 11.3 Exploratory anatomical interpretation

Only selected examinations and imaging slices have undergone detailed anatomical review.

Not every false-positive voxel has been assigned to a specific anatomical structure.

### 11.4 Hypothesised error mechanisms

The observed association between heterogeneous tumour tissue and false-positive RC predictions does not prove that necrosis causes the misclassification.

Similarly, the apparent association with normal CSF spaces does not establish that the model relies primarily on CSF-like signal intensity.

These hypotheses require further testing.

### 11.5 Post-processing selection bias

The ET–RC exclusion intervention was proposed after inspecting the validation errors.

The same validation subset was then used to measure its effectiveness.

The resulting improvement may therefore be optimistic.

### 11.6 Potential sensitivity loss

ET–RC exclusion removes both false-positive and true-positive cavity voxels.

The resulting reduction in recall may be clinically important, particularly in selected examinations.

### 11.7 No clinical validation

The model has not undergone external validation, prospective evaluation or clinical deployment testing.

It should not be used to guide clinical decisions.

---

## 12. Future Experiments

### 12.1 Detailed review of deteriorating cases

Examinations demonstrating reduced Dice or recall after ET–RC exclusion should undergo further anatomical review.

Priority cases include:

- `BraTS-GLI-02817-100`
- `BraTS-GLI-02158-101`
- `BraTS-GLI-02260-100`

The aim is to determine whether the correction removes correctly segmented cavity tissue because of local image characteristics, overlapping predicted ET or other factors.

### 12.2 Anatomically informed hard-negative sampling

Future training experiments may include additional negative examples involving:

- Normal ventricular CSF spaces.
- Sulci and fissures.
- Heterogeneous tumour-associated abnormalities.
- Structures resembling postoperative cavities.

Hard-negative examples should be drawn from the training cohort rather than selected from validation data for training.

### 12.3 Improved segmentation consistency

Potential approaches include:

- Alternative output representations.
- Explicit ET–RC consistency penalties.
- Hierarchical or mutually exclusive tissue classification.
- More selective post-processing rules.

These approaches should be compared with the existing baseline using controlled experiments.

### 12.4 Greater anatomical context

Alternative patch sizes or architectures may be investigated to determine whether increased anatomical context reduces CSF-space-associated errors.

Such experiments must consider GPU memory constraints.

### 12.5 Expanded validation

A larger validation cohort should be evaluated to determine whether the observed error patterns and post-processing improvements remain consistent.

### 12.6 Improved evaluation criteria

Future evaluation should include:

- Region-specific Dice, precision and recall.
- False-positive voxel volume in region-absent examinations.
- Case-level segmentation performance.
- ET–RC prediction consistency.
- Anatomical review of clinically important errors.
- Potentially surface-distance metrics and lesion-level measures.

### 12.7 Independent held-out testing

The held-out test cohort should remain reserved for later independent assessment.

Final model selection and post-processing decisions should be completed before evaluating the test cohort.

---

## 13. Longer-Term Research Directions

### 13.1 Three-dimensional visualisation and printing

Explore conversion of MRI segmentation outputs into three-dimensional anatomical models.

Potential applications include:

- Integration with 3D Slicer.
- Three-dimensional visualisation of tumour-associated abnormalities.
- Preparation of printable anatomical models.
- Exploratory applications in surgical planning and education.

### 13.2 Longitudinal postoperative assessment

Investigate quantitative changes in tumour-associated regions across serial MRI examinations.

### 13.3 Tumour recurrence research

Explore whether segmentation-derived imaging features could contribute to studies of postoperative tumour recurrence.

Potential future extensions include integration with diffusion tensor imaging and other advanced MRI techniques.

These directions remain exploratory and have not been demonstrated by the current baseline model.

---

## 14. Reproducibility and Repository Organisation

### 14.1 Core implementation

Relevant files include:

- `src/preprocessing.py`
- `src/losses.py`
- `notebooks/01_mri_data_exploration.ipynb`
- `notebooks/03_unet_model_development.ipynb`

### 14.2 Experiment records

The completed baseline experiment is:

`unet_180train_24val_20261013_203139`

Important local experiment files include:

- `logs/training_history.csv`
- `logs/validation_region_history.csv`
- `logs/validation_case_metrics_best_epoch38.csv`
- `logs/validation_region_summary_best_epoch38.csv`
- `logs/validation_et_rc_overlap_best_epoch38.csv`
- `logs/rc_et_exclusion_comparison_best_epoch38.csv`
- `checkpoints/best_model.pt`
- `checkpoints/final_latest.pt`

Large imaging datasets and model checkpoints are stored outside the public repository.

### 14.3 Reproducibility decisions

The project includes:

- Subject-level splitting to reduce data leakage.
- A fixed dataset split seed.
- Deterministic validation preprocessing.
- Explicit experiment configuration.
- Saved checkpoints.
- Training and validation histories.
- Re-evaluation of the best checkpoint.
- Case-level error analysis.
- Preservation of baseline and post-processing results.

---

## 15. Current Project Status

### Completed

- BraTS 2024 GLI dataset acquisition.
- Dataset inventory and subject identification.
- Subject-level train/validation/test splitting.
- Multimodal MRI preprocessing.
- Training augmentation pipeline.
- Four-region segmentation label conversion.
- Baseline 3D U-Net implementation.
- Combined Dice and BCE loss.
- Forty-epoch baseline training.
- Checkpoint preservation and verification.
- Region-specific validation evaluation.
- Case-level RC false-positive analysis.
- Anatomical identification of Sylvian fissure misclassification.
- Investigation of tumour-associated RC misclassification.
- ET–RC prediction-overlap analysis.
- ET–RC exclusion post-processing experiment.
- Case-level improvement and deterioration analysis.
- Exploratory cavity-size analysis.

### Pending

- Detailed MRI review of cases worsened by ET–RC exclusion.
- Additional validation on a larger cohort.
- Controlled model-improvement experiments.
- Independent held-out test evaluation.
- Publication-quality MRI figures.
- Final project README and navigation updates.
- References and citation verification.

---

## 16. Preliminary Conclusions

A baseline 3D U-Net was successfully trained to segment four glioma-associated regions using multimodal MRI.

The best checkpoint achieved a validation macro Dice of approximately **0.681** on a limited 24-examination validation subset.

However, detailed evaluation revealed substantial false-positive resection-cavity predictions, particularly in examinations without annotated cavities.

Clinical anatomical review identified two principal error patterns:

1. Normal CSF spaces misclassified as postoperative cavities.
2. Heterogeneous tumour-associated tissue misclassified as cavities.

Additional analysis demonstrated frequent simultaneous ET–RC predictions, particularly in RC-absent examinations.

A simple ET–RC exclusion rule reduced false-positive RC volume by **86.55%** in RC-absent examinations and improved mean RC Dice from **0.5405 to 0.5748** in RC-present examinations.

However, the correction also reduced recall and worsened segmentation in four RC-present examinations.

These findings demonstrate that an intervention can improve average segmentation metrics while still introducing clinically relevant errors.

**The central methodological conclusion is that quantitative segmentation evaluation should be complemented by clinically informed anatomical error analysis.**

The next stage of the project will investigate the causes of post-processing deterioration and evaluate whether more selective or learned approaches can improve anatomical consistency without compromising genuine cavity detection.

---

## 17. References

*To be completed before final publication.*

Required references include:

- Official BraTS 2024 Glioma dataset documentation and citation.
- Original U-Net and relevant 3D U-Net methodology papers.
- MONAI framework documentation or publication.
- Relevant literature on postoperative glioma segmentation.
- Research on anatomical consistency in medical image segmentation.
- Literature supporting future work on glioma recurrence prediction and advanced MRI.

---

**Document status:** Living research report. Results will be updated as additional validation, experiments and independent testing are completed.
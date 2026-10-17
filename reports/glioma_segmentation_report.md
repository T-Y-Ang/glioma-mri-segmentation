# Glioma MRI Segmentation — Research Report

**Project:** Deep Learning for Glioma MRI Segmentation
**Dataset:** BraTS 2024 Glioma (GLI)
**Framework:** PyTorch and MONAI
**Model:** 3D U-Net
**Status:** Experimental research completed; independent held-out evaluation completed; repository finalisation in progress
**Last updated:** 17 October 2026

> **Research status:** This living research report documents the completed model-development experiments, validation analyses, independent held-out evaluation, clinical anatomical observations, limitations and future directions. The findings represent retrospective evaluation within a single benchmark dataset and do not establish clinical deployment readiness.


---

## 1. Project Overview and Clinical Motivation

### 1.1 Background

Glioma segmentation is an important component of neuro-oncological imaging research. Delineation of tumour-associated abnormalities and postoperative resection cavities may support quantitative assessment of disease extent, longitudinal imaging analysis and future development of computer-assisted surgical planning tools [1, 2].

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

The project uses the **BraTS 2024 Glioma (GLI)** training dataset obtained through Synapse [1, 2].

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

The experimental subset sizes refer to MRI examinations, not unique subjects. The larger dataset partitions were established using subject-level separation to prevent cross-partition leakage. The model-development subsets were drawn from their respective partitions.

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

Following model development and validation-stage experiments, the selected baseline checkpoint was evaluated on the complete held-out test partition of **198 examinations from 92 subjects**. The independent evaluation was conducted using a protocol frozen before test results were examined. Full results are presented in Section 10.

---

## 3. Segmentation Regions and Label Representation

### 3.1 Original reference labels

The original segmentation labels are [1, 2]:

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

A 3D U-Net was implemented using MONAI [3, 4].

| Parameter | Configuration |
|---|---|
| Architecture | 3D U-Net |
| Input channels | 4 |
| Output channels | 4 |
| Encoder channels | 16, 32, 64, 128, 256 |
| Strides | 2, 2, 2, 2 |
| Residual units per stage | 2 |
| Trainable parameters | 4,811,129 |
| Output activation | Sigmoid applied to model logits during inference |

The four input channels correspond to the MRI modalities.

The four output channels correspond to WT, TC, ET and RC.

### 5.2 Loss function

The training objective combines Dice loss [5] and binary cross-entropy loss.

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

These initial validation findings informed subsequent clinical error analysis and experimental comparisons. The selected baseline checkpoint was subsequently evaluated on the independent held-out test cohort, as reported in Section 10.

### 6.5 Cavity-Aware Sampling Experiment

Following baseline training and clinical error analysis, a second training experiment investigated whether increasing the representation of resection-cavity-containing patches could improve cavity segmentation.

The experiment was identified as:

`unet_cavity_aware_180train_24val_20261016_042611`

The experiment retained the same general 3D U-Net architecture and model-development cohort, allowing comparison with the original baseline while investigating a modified sampling strategy.

Training was completed over 40 epochs.

The cavity-aware experiment achieved a best validation macro Dice of approximately **0.6740**, with a corresponding resection-cavity Dice of approximately **0.5212**.

By comparison, the original baseline achieved a best validation macro Dice of **0.6809** and resection-cavity Dice of **0.5405**.

The alternative sampling strategy therefore did not demonstrate an improvement in either overall validation macro Dice or mean resection-cavity Dice.

Furthermore, the cavity-aware experiment produced an increased burden of false-positive cavity predictions during validation.

These findings suggest that increasing exposure to cavity-containing training patches, without adequately addressing difficult negative examples and anatomical discrimination, may be insufficient to improve cavity segmentation.

However, this was a limited experimental comparison rather than a comprehensive assessment of cavity-aware sampling strategies.

### 6.6 Final Model Selection

The original baseline checkpoint from **epoch 38** was retained for independent evaluation.

This decision was based on its validation performance and the absence of a convincing improvement from the cavity-aware experiment.

The selected checkpoint, raw prediction threshold and validation-informed relative-confidence post-processing method were recorded in the frozen evaluation protocol before the held-out test cohort was examined.

No further training or model selection was performed using the held-out test results.

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

The following results were obtained during the initial ET-priority post-processing experiment. Minor one-voxel differences from the independently reproduced validation metrics in Section 7.4 reflect differences observed between inference runs. The original recorded measurements are retained rather than retrospectively harmonised.

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

The unconditional ET-priority intervention was developed and assessed on the limited validation subset. It was not selected for the independent held-out test comparison. Subsequent validation experiments led to selection of the relative-confidence method as the secondary test analysis, as described in Section 9.9.

### 9.9 Relative-Confidence Post-processing Experiment

The following values were obtained during the subsequent three-method comparison using the original, ET-priority and relative-confidence predictions. The raw baseline count of 102,424 differs by one voxel from the earlier post-processing experiment. This discrepancy was observed between evaluation runs and has not been attributed to a confirmed technical cause.

#### Rationale

The initial ET-priority exclusion rule substantially reduced false-positive resection-cavity predictions but also removed genuine cavity tissue in some examinations.

In examination `BraTS-GLI-02817-100`, ET-priority removed 193 correctly predicted cavity voxels, reducing recall by 18.45 percentage points.

Further investigation demonstrated that some genuine cavity voxels were assigned high probabilities for both enhancing tumour (ET) and resection cavity (RC).

Among the 193 removed true-positive voxels, mean predicted ET and RC probabilities were 0.8862 and 0.8591, respectively. Eighty reference cavity voxels had both ET and RC scores of at least 0.9.

These findings motivated an alternative post-processing strategy.

#### Method

A relative-confidence rule was evaluated using the same epoch-38 model checkpoint.

When both ET and RC were predicted, RC was retained if its sigmoid score was greater than or equal to the ET score.

The rule was compared against:

1. The original RC prediction.
2. Unconditional ET-priority exclusion.
3. Relative-confidence priority.

All 24 validation examinations were evaluated without retraining.

#### Results in RC-present examinations

| Metric | Original | ET-priority | Relative-confidence |
|---|---:|---:|---:|
| Mean Dice | 0.5405 | 0.5748 | 0.5754 |
| Mean precision | 0.5377 | 0.5906 | 0.5878 |
| Mean recall | 0.6490 | 0.6261 | 0.6347 |

Relative-confidence priority achieved a marginally higher mean Dice than ET-priority, with a difference of 0.0006.

Relative-confidence also increased recall compared with ET-priority but reduced precision.

#### Results in RC-absent examinations

| Method | False-positive RC voxels |
|---|---:|
| Original | 102,424 |
| ET-priority | 13,777 |
| Relative-confidence | 15,886 |

Compared with the original model, ET-priority reduced false-positive RC volume by 86.55%, while relative-confidence reduced it by 84.49%.

Both methods retained false-positive predictions in all five RC-absent examinations.

#### Paired comparison

Comparing relative-confidence directly against ET-priority in the 19 RC-present examinations:

- Dice improved in eight examinations, worsened in eight and remained unchanged in three.
- Precision improved in four, worsened in 12 and remained unchanged in three.
- Recall improved in 14, worsened in none and remained unchanged in five.
- The relative-confidence rule restored 978 true-positive RC voxels but also restored 3,397 false-positive voxels.

Compared with the original model, ET-priority improved Dice in 13 examinations and worsened it in four. Relative-confidence improved Dice in 15 examinations and worsened it in two.

However, improvements were unevenly distributed.

In `BraTS-GLI-02382-100`, relative-confidence increased Dice from 0.2360 to 0.5108. This single examination accounted for approximately 41% of the mean Dice improvement across the 19 RC-present examinations.

#### Interpretation

Relative-confidence priority partially mitigated the sensitivity loss caused by unconditional ET-priority exclusion.

However, it also restored additional false-positive predictions and did not consistently improve Dice compared with ET-priority.

The marginal difference in mean Dice does not establish superiority, particularly given the small validation sample and the exploratory development of both methods on the same cohort.

Neither method resolved the normal CSF-space-associated errors identified during anatomical review.

These findings suggest that additional post-processing rules alone may be insufficient to address the model's underlying difficulties in distinguishing enhancing tumour, postoperative cavity and normal CSF spaces.

Future experiments should investigate training-based approaches to improving tissue discrimination and anatomical consistency.

Following exploratory comparison on the validation subset, the relative-confidence RC post-processing rule was selected as the pre-specified secondary analysis for independent held-out testing. The unmodified baseline predictions were retained as the primary analysis. Both evaluation methods were frozen before the test cohort was examined.

---

## 10. Independent Held-Out Test Evaluation

### 10.1 Evaluation Cohort and Frozen Protocol

The selected baseline 3D U-Net was evaluated on an independent held-out cohort comprising **198 MRI examinations from 92 subjects**. Subject-level separation was verified, with no overlap between the training, validation and test cohorts.

The baseline model checkpoint was selected at epoch 38 based on validation performance. Although a subsequent cavity-aware sampling experiment was completed, the original baseline checkpoint was retained for final evaluation.

The evaluation protocol was recorded before examining the held-out test results.

Two analyses were specified:

1. **Primary analysis — raw baseline predictions:** Independent sigmoid outputs thresholded at 0.5, without anatomical post-processing.
2. **Secondary analysis — relative-confidence correction:** A validation-informed rule that modifies only the resection-cavity (RC) prediction when RC and enhancing tumour (ET) are simultaneously predicted.

The secondary correction retains an RC voxel if ET is not predicted at that location or if the predicted RC probability is greater than or equal to the predicted ET probability.

Full-volume inference used MONAI sliding-window inference with a 96 × 96 × 96 voxel region of interest, 25% overlap and a sliding-window batch size of one.

No model retraining, threshold optimisation or post-processing adjustment was performed using the held-out test results.

### 10.2 Regional Segmentation Performance

The selected baseline model was evaluated across four segmentation regions: whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC).

Mean Dice, precision and recall were calculated among examinations in which the corresponding reference region was present.

For each segmentation region, mean Dice was calculated across examinations containing that region in the reference annotation. The macro-average Dice was then calculated as the unweighted arithmetic mean of the four regional mean Dice values (WT, TC, ET and RC).

Consequently, each region contributed equally to the macro-average, regardless of the number of reference-present examinations. Region-absent examinations were excluded from the corresponding regional Dice mean and were evaluated separately using false-positive measurements.

This definition differs from calculating a single mean Dice across all examination–region combinations.

| Region | Reference-present examinations | Raw Dice | Corrected Dice | Raw precision | Corrected precision | Raw recall | Corrected recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| WT | 198 | 0.8126 | 0.8126 | 0.8147 | 0.8147 | 0.8372 | 0.8372 |
| TC | 151 | 0.6298 | 0.6298 | 0.7124 | 0.7124 | 0.6457 | 0.6457 |
| ET | 151 | 0.6456 | 0.6456 | 0.6946 | 0.6946 | 0.6784 | 0.6784 |
| RC | 157 | 0.5221 | **0.5638** | 0.4872 | **0.5672** | 0.6652 | 0.6470 |

The macro-average Dice across the four regions was **0.6525** for the raw baseline and **0.6630** following relative-confidence correction.

Relative-confidence correction increased mean RC Dice by 0.0417 and mean RC precision by approximately 0.0800, while reducing mean RC recall by approximately 0.0182.

The WT, TC and ET results remained unchanged because the post-processing rule modified only the RC channel.

The raw test macro Dice of 0.6525 was lower than the baseline validation macro Dice of 0.6809. This difference should be interpreted cautiously because the validation evaluation involved only 24 examinations, whereas the held-out test cohort contained 198 examinations.

### 10.3 False-Positive Predictions in Region-Absent Examinations

Regional Dice scores calculated only in reference-present examinations do not fully describe false-positive segmentation in examinations where a region is absent.

The held-out test cohort contained:

| Region | Reference-absent examinations | Examinations with false-positive predictions | Raw false-positive voxels | Corrected false-positive voxels |
|---|---:|---:|---:|---:|
| WT | 0 | 0 | 0 | 0 |
| TC | 47 | 36 | 21,055 | 21,055 |
| ET | 47 | 36 | 20,172 | 20,172 |
| RC | 41 | 41 | 496,653 | **147,885** |

Relative-confidence correction reduced the total false-positive RC voxel burden in cavity-absent examinations by **348,768 voxels (70.2%)**.

However, all 41 cavity-absent examinations retained some false-positive RC predictions after correction.

The intervention therefore reduced the extent of erroneous cavity segmentation but did not eliminate false-positive cavity detection.

These findings reinforce the importance of reporting both region-present segmentation metrics and false-positive predictions in region-absent examinations.

### 10.4 Paired Resection-Cavity Analysis

To assess the consistency of relative-confidence post-processing, examination-level RC segmentation performance was compared before and after correction.

The paired analysis was restricted to **157 held-out test examinations containing an annotated resection cavity**. Both methods were evaluated against identical reference segmentations, allowing within-examination comparison without confounding from differences in case composition.

#### 10.4.1 Distribution of examination-level outcomes

| Outcome | Examinations | Percentage |
|---|---:|---:|
| RC Dice improved | 111 | 70.7% |
| RC Dice deteriorated | 22 | 14.0% |
| RC Dice unchanged | 24 | 15.3% |
| **Total** | **157** | **100.0%** |

Relative-confidence correction improved RC Dice in approximately seven out of ten cavity-present examinations.

However, the improvement was not universal, and 22 examinations demonstrated deterioration.

#### 10.4.2 Magnitude and distribution of Dice changes

| Measurement | Result |
|---|---:|
| Mean paired RC Dice change | +0.0417 |
| Median paired RC Dice change | +0.0015 |
| Greatest individual improvement | +0.4123 |
| Greatest individual deterioration | −0.5299 |

The positive mean Dice change indicates an overall improvement in segmentation agreement with the reference cavity annotations.

However, the median improvement was substantially smaller than the mean, suggesting that the aggregate benefit was influenced by a subset of examinations with relatively large improvements.

This distinction is important because a positive average result does not imply that the intervention produces a substantial benefit in a typical examination.

#### 10.4.3 Effect on true-positive and false-positive cavity voxels

Across the 157 cavity-present examinations, relative-confidence correction removed:

| Removed voxel category | Count |
|---|---:|
| False-positive RC voxels | 520,276 |
| True-positive RC voxels | 21,738 |
| **Total removed** | **542,014** |

Approximately **96.0% of the removed voxels were false positives**, while approximately 4.0% were true positives relative to the reference annotations.

The predominance of false-positive removal explains the improvement in mean RC precision and Dice.

Nevertheless, removal of true-positive cavity voxels reduced mean recall and contributed to deterioration in some examinations.

These figures describe voxel-level agreement with the reference masks; they do not independently establish the anatomical correctness of every removed voxel.

#### 10.4.4 Clinical implications of heterogeneous responses

The paired results demonstrate that relative-confidence correction is effective at suppressing a substantial proportion of erroneous cavity predictions, but its effect varies considerably between examinations.

The intervention is particularly relevant when false-positive cavity predictions overlap regions assigned greater ET probability by the model.

Conversely, genuine annotated cavity voxels may be removed when the model assigns greater ET than RC probability, even if the original RC prediction was correct relative to the reference segmentation.

The greatest individual deterioration occurred in examination `BraTS-GLI-02119-101`, where RC Dice decreased from 0.5345 to 0.0046. This examination underwent subsequent post hoc multisequence anatomical review, described in Section 10.6.

Overall, the paired analysis supports a positive average effect of relative-confidence post-processing while highlighting the need to examine the distribution of individual outcomes, rather than relying solely on aggregate Dice scores.

### 10.5 Subject-Clustered Bootstrap Analysis

#### 10.5.1 Rationale

The held-out test cohort contained 198 MRI examinations from 92 subjects, with some subjects contributing multiple examinations.

Repeated examinations from the same subject may share anatomical, pathological and imaging characteristics. Consequently, treating all examinations as statistically independent could underestimate uncertainty in the estimated effect of post-processing.

A **subject-clustered bootstrap analysis** was therefore performed to estimate the uncertainty surrounding the mean paired improvement in resection-cavity (RC) Dice.

The analysis focused on examinations containing an annotated resection cavity, because regional Dice was defined and reported for reference-present examinations.

#### 10.5.2 Methodology

The analysis included **157 RC-present examinations from 78 unique subjects**.

For each examination, the paired difference in RC Dice was calculated as:

\[
\Delta \mathrm{Dice}_i =
\mathrm{Dice}_{i,\mathrm{corrected}}
-
\mathrm{Dice}_{i,\mathrm{raw}}
\]

A positive value indicated improved agreement with the reference cavity segmentation following relative-confidence correction.

The bootstrap procedure consisted of:

1. Sampling subjects with replacement from the 78 eligible subjects.
2. Retaining all corresponding RC-present examinations for each sampled subject, including repeated contributions when a subject was selected more than once.
3. Calculating the mean paired RC Dice difference across the resampled examinations.
4. Repeating the procedure for **10,000 bootstrap iterations**.

A fixed random seed of 42 was used to support reproducibility.

The 95% confidence interval was estimated using the 2.5th and 97.5th percentiles of the bootstrap distribution.

This approach preserved the clustering of examinations within subjects during resampling.

#### 10.5.3 Results

| Measurement | Result |
|---|---:|
| RC-present examinations | 157 |
| Unique subjects | 78 |
| Bootstrap iterations | 10,000 |
| Random seed | 42 |
| Observed mean paired RC Dice improvement | +0.0417 |
| 95% subject-clustered bootstrap confidence interval | **+0.0227 to +0.0625** |

The observed mean paired improvement in RC Dice was **+0.0417**, with a 95% subject-clustered bootstrap confidence interval of **+0.0227 to +0.0625**.

The confidence interval did not include zero, indicating that the estimated average improvement remained positive after accounting for within-subject clustering through bootstrap resampling.

#### 10.5.4 Interpretation

The subject-clustered bootstrap analysis supports a positive average improvement in RC Dice following relative-confidence post-processing within the evaluated held-out cohort.

Importantly, the confidence interval quantifies uncertainty in the **mean paired difference**, rather than the probability that an individual examination will improve.

The result therefore does not imply that the correction is beneficial in every examination. As demonstrated in Section 10.4, some examinations experienced substantial deterioration despite the positive average effect.

Furthermore, the confidence interval does not establish clinical superiority, anatomical correctness or generalisability to external datasets.

The post-processing method was developed using the validation cohort and specified before held-out testing. However, the bootstrap analysis remains an internal statistical assessment of performance on a single dataset.

Taken together, the paired examination-level results and subject-clustered bootstrap analysis provide evidence that relative-confidence correction improves average reference-based RC segmentation performance, while highlighting the importance of case-level variability and independent clinical assessment.

### 10.6 Clinical Review of a Potential Resection-Cavity Annotation Disagreement

A detailed qualitative review was undertaken for examination `BraTS-GLI-02119-101`, which demonstrated the greatest deterioration in resection-cavity Dice following relative-confidence post-processing (0.5345 to 0.0046).

Multisequence MRI review identified a heterogeneous region of substantial post-contrast T1 enhancement extending across multiple adjacent axial slices. Much of this region was assigned to the resection-cavity class in the reference segmentation, while the reference enhancing-tumour annotation was predominantly peripheral.

The baseline model predicted enhancing tumour within 14,444 of 15,600 voxels (92.59%) annotated as resection cavity. Within these discordant voxels, mean predicted ET and RC probabilities were 0.9694 and 0.4771, respectively, with ET probability exceeding RC probability in 99.88%.

These findings suggest a substantial disagreement between the model's segmentation and the reference annotation. The imaging appearance raises the possibility that the annotated region includes enhancing postoperative tissue rather than exclusively fluid-containing cavity space. However, enhancement alone cannot distinguish viable tumour from postoperative reactive changes, and model confidence does not independently establish tissue identity.

Accordingly, this examination is classified as a **potential annotation ambiguity requiring expert adjudication**, rather than a confirmed annotation error. The original reference labels and held-out test metrics were preserved without modification.

This case illustrates a limitation of annotation-based segmentation benchmarks: disagreement with a reference mask does not necessarily establish clinical incorrectness, particularly in complex postoperative anatomy.

### 10.7 Interpretation of Independent Test Results

The independent held-out evaluation demonstrated that a baseline 3D U-Net trained on a relatively small subset of the BraTS 2024 GLI dataset could achieve meaningful segmentation agreement across four postoperative glioma-associated regions. However, performance varied substantially between anatomical regions, and resection-cavity segmentation remained particularly challenging.

Five principal findings emerged.

#### 10.7.1 Generalisation of the baseline model

The baseline model achieved a macro-average Dice of **0.6525** across 198 held-out examinations from 92 subjects, compared with 0.6809 on the 24-examination validation subset.

Whole-tumour segmentation demonstrated the strongest performance, achieving a mean Dice of 0.8126. Tumour-core and enhancing-tumour segmentation achieved intermediate performance, while resection-cavity segmentation remained the weakest region, with a mean Dice of 0.5221.

These results demonstrate that the model learned transferable segmentation features, although the reduction in performance between validation and testing suggests that the initial validation results may not fully represent performance across the broader dataset.

The relatively small training subset, heterogeneous postoperative imaging appearances and limited anatomical context available during patch-based training may have contributed to the observed performance limitations. These remain plausible explanations rather than experimentally established causes.

#### 10.7.2 Effectiveness of relative-confidence post-processing

Relative-confidence correction improved mean resection-cavity Dice from **0.5221 to 0.5638**, increasing the overall macro-average Dice from 0.6525 to 0.6630.

The improvement was accompanied by increased RC precision and reduced RC recall, consistent with the removal of both false-positive and true-positive cavity voxels.

In examinations without annotated resection cavities, the correction reduced false-positive RC volume by **70.2%**, demonstrating substantial suppression of erroneous cavity predictions.

Nevertheless, all 41 cavity-absent examinations retained some false-positive RC predictions after correction.

The intervention therefore improved reference-based segmentation performance but did not resolve the underlying difficulty of distinguishing postoperative cavities from other intracranial structures.

#### 10.7.3 Heterogeneity of examination-level responses

Relative-confidence correction improved RC Dice in 111 of 157 cavity-present examinations, corresponding to 70.7% of the eligible test cohort.

However, 22 examinations deteriorated, and 24 remained unchanged.

The mean paired Dice improvement was +0.0417, whereas the median improvement was only +0.0015. This discrepancy indicates that the average improvement was influenced by a subset of examinations with relatively large gains.

The subject-clustered bootstrap analysis produced a 95% confidence interval of **+0.0227 to +0.0625** for the mean paired improvement, supporting a positive average effect within the evaluated cohort.

These findings establish that relative-confidence correction can improve average segmentation agreement without providing uniformly beneficial results across individual examinations.

#### 10.7.4 Clinical and anatomical limitations

Clinical error analysis identified distinct patterns of resection-cavity misclassification.

First, normal CSF-containing anatomical structures, including the Sylvian fissure and probable ventricular spaces, were sometimes misclassified as postoperative cavities. These errors were not consistently addressed by ET–RC relative-confidence correction because they did not necessarily involve simultaneous enhancing-tumour predictions.

Second, heterogeneous tumour-associated abnormalities were frequently associated with false-positive RC predictions. These errors were more amenable to correction when the model simultaneously predicted ET with greater confidence than RC.

Third, the held-out examination `BraTS-GLI-02119-101` demonstrated a substantial disagreement between the model's predicted ET region and the reference RC annotation. Relative-confidence correction produced a marked deterioration in reference-based RC Dice despite the presence of heterogeneous post-contrast enhancement on multisequence anatomical review.

This examination raises the possibility of reference-annotation ambiguity, but neither an annotation error nor the presence of viable tumour can be established without further expert assessment.

These observations highlight the limitations of using numerical segmentation overlap as the sole measure of model quality. Agreement with a reference annotation is an important benchmark, but it does not independently establish anatomical correctness or clinical validity.

#### 10.7.5 Implications for future model development

The independent test findings suggest that future improvements should address the model's underlying representation of postoperative anatomy rather than relying exclusively on increasingly complex post-processing rules.

Potential research directions include:

1. **Improved anatomical context:** Investigating larger receptive fields or alternative architectures that better distinguish normal CSF spaces from postoperative cavities.
2. **Anatomically consistent segmentation:** Developing output representations or training objectives that account for the mutually exclusive nature of reference ET and RC annotations while preserving the nested relationships between ET, TC and WT.
3. **Improved sampling strategies:** Investigating clinically informed hard-negative sampling and more representative training cohorts.
4. **Reference-annotation assessment:** Evaluating ambiguous postoperative regions through independent expert review and, where available, additional imaging or longitudinal information.
5. **Expanded evaluation:** Incorporating surface-distance measures, lesion-level analysis and external validation to better characterise clinically relevant segmentation errors.

These directions remain proposals for future research and were not evaluated as part of the frozen held-out test protocol.

#### 10.7.6 Overall interpretation

The independent evaluation supports three overarching conclusions.

**First**, a conventional 3D U-Net can produce meaningful multimodal postoperative glioma segmentations using consumer-grade computing hardware, although performance remains variable across anatomical regions.

**Second**, validation-informed relative-confidence post-processing provides a measurable improvement in average reference-based resection-cavity segmentation performance, supported by subject-clustered uncertainty estimation. However, this improvement is heterogeneous and does not eliminate false-positive cavity predictions or occasional severe deterioration.

**Third**, clinical anatomical review provides information that cannot be obtained from aggregate segmentation metrics alone. The identification of normal CSF-space misclassification and potential postoperative annotation ambiguity demonstrates the value of integrating clinical expertise into model evaluation.

The project therefore illustrates a reproducible approach to combining deep learning, quantitative performance assessment and clinically informed error analysis.

The results represent retrospective benchmark performance and should not be interpreted as evidence of clinical deployment readiness.

## 11. Integrated Clinical Interpretation

### 11.1 Distinct Anatomical Error Patterns Require Different Solutions

Clinical review of the model's segmentation predictions identified two principal patterns of resection-cavity (RC) misclassification.

**Normal CSF-space-associated errors** involved intracranial structures that may share fluid-related MRI characteristics with postoperative cavities. In examination `BraTS-GLI-02736-101`, false-positive RC predictions were anatomically localised to the Sylvian fissure. Another examination, `BraTS-GLI-02609-101`, demonstrated probable ventricular CSF-space-associated misclassification.

These findings suggest that the model may have difficulty distinguishing normal CSF-containing anatomy from postoperative cavities, potentially because of similarities in local imaging characteristics or insufficient anatomical context.

**Tumour-associated errors** involved heterogeneous tumour-related abnormalities that were incorrectly predicted as RC. In several validation examinations, these false-positive predictions overlapped reference enhancing-tumour (ET) regions and were associated with simultaneous ET–RC predictions.

The distinction between these error patterns became important during post-processing development.

Relative-confidence correction preferentially removed RC predictions where the model also predicted ET with greater confidence. This approach substantially reduced tumour-associated false-positive cavity predictions but did not directly address false positives arising in normal CSF spaces without competing ET predictions.

The persistence of false-positive RC predictions across all 41 cavity-absent held-out test examinations further demonstrates that the post-processing intervention did not eliminate the underlying segmentation errors.

These observations suggest that **different anatomical error mechanisms may require different corrective strategies**, rather than a single universal post-processing rule.

### 11.2 Anatomical Consistency Versus Segmentation Accuracy

The model was trained to predict four overlapping binary segmentation regions using independent sigmoid output channels.

The tumour-associated regions have a hierarchical relationship:

**ET ⊆ TC ⊆ WT**

Resection cavity is represented separately and is mutually exclusive with ET in the reference annotations.

However, the model architecture does not explicitly enforce these relationships. Consequently, simultaneous ET and RC predictions may occur at individual voxels.

The relative-confidence correction was designed to resolve one form of this inconsistency by retaining RC predictions only when the model did not simultaneously predict ET with greater probability.

On the independent held-out test cohort, this intervention improved mean RC Dice from **0.5221 to 0.5638** and reduced false-positive RC volume by **70.2%** in cavity-absent examinations.

Nevertheless, it also removed correctly predicted reference cavity voxels, reducing recall and worsening Dice in 22 of 157 cavity-present examinations.

This illustrates a distinction between **enforcing consistency between predicted labels** and **achieving accurate anatomical segmentation**.

A prediction may become internally consistent while losing agreement with the reference segmentation. Conversely, a prediction that violates a simple anatomical rule may still contain voxels that agree with the reference annotation.

Furthermore, the independent sigmoid scores are not necessarily calibrated or directly comparable as probabilities of true tissue identity. Their relative magnitudes provide a practical post-processing criterion but do not establish anatomical or pathological correctness.

Thus, anatomically motivated post-processing should be evaluated through both aggregate metrics and examination-level review.

### 11.3 Reference-Annotation Uncertainty in Postoperative Imaging

Postoperative glioma imaging presents challenges that extend beyond conventional tumour segmentation.

The operative bed may contain fluid, blood products, surgically altered tissue, postoperative enhancement and residual tumour-associated abnormalities. These structures may demonstrate overlapping MRI appearances, particularly when only standard structural sequences are available.

The held-out examination `BraTS-GLI-02119-101` illustrates this difficulty.

Following relative-confidence correction, RC Dice decreased markedly from **0.5345 to 0.0046**. Subsequent post hoc review demonstrated substantial heterogeneous post-contrast enhancement across multiple adjacent slices within a region predominantly annotated as resection cavity.

The model assigned ET predictions to **92.59% of the reference RC voxels**, with ET probability exceeding RC probability in 99.88% of these discordant voxels.

This represents a substantial disagreement between the model's interpretation and the reference segmentation.

However, enhancement does not necessarily indicate viable tumour. Postoperative reactive changes, surgical injury and other non-neoplastic processes may also produce enhancement.

Likewise, high model confidence does not independently establish the identity of the underlying tissue.

The examination therefore raises the possibility of **reference-annotation ambiguity**, but it does not demonstrate that the annotation is incorrect or that the model is clinically superior.

Independent expert adjudication, supported where appropriate by additional imaging and longitudinal assessment, would be required to investigate this possibility.

The original reference labels and held-out test results were preserved without modification.

This case illustrates a fundamental limitation of reference-based segmentation evaluation: **numerical disagreement with an annotation does not necessarily establish clinical incorrectness**, particularly when the underlying tissue classification is itself uncertain.

### 11.4 Contribution of Clinical Expertise to Model Evaluation

The initial quantitative evaluation identified resection-cavity segmentation as the weakest-performing region.

However, numerical metrics alone did not explain the anatomical distribution of errors.

Clinical review of multimodal MRI and segmentation overlays identified the Sylvian fissure as a source of false-positive cavity predictions, distinguished normal CSF-space-associated errors from tumour-associated misclassification, and highlighted potential uncertainty in a postoperative reference annotation.

These observations informed different stages of the investigation.

Recognition of tumour-associated false-positive RC predictions motivated the analysis of simultaneous ET–RC outputs and subsequent post-processing experiments.

Recognition of normal CSF-space misclassification demonstrated a limitation of those interventions, because removing overlapping ET–RC predictions does not necessarily correct errors involving normal CSF anatomy.

Finally, multisequence review of the severely deteriorating held-out examination demonstrated why a large reduction in reference-based Dice should prompt anatomical investigation rather than automatic assumptions about the clinical correctness of either the model or the annotation.

These findings illustrate the value of incorporating clinical expertise into machine-learning research, particularly when evaluating complex postoperative neuroimaging.

Clinical interpretation can help identify meaningful error patterns, generate hypotheses for model improvement and recognise limitations that are not apparent from aggregate performance statistics.

Nevertheless, qualitative review remains partly subjective and should not be treated as a substitute for independent expert annotation or systematic clinical validation.

### 11.5 Implications for Clinically Meaningful Model Development

The combined validation and held-out test findings suggest that future segmentation research should address three related challenges.

**First, anatomical context.** The model must distinguish structures that may share local MRI signal characteristics but differ in anatomical location and clinical significance. Larger receptive fields, alternative architectures and anatomically informed training strategies may improve this distinction.

**Second, segmentation consistency.** The model should better represent the hierarchical relationships between tumour-associated regions and the distinction between enhancing tumour and resection cavity. Explicitly constrained outputs or alternative label representations may reduce contradictory predictions.

**Third, uncertainty in reference annotations.** Postoperative segmentation benchmarks should be interpreted with awareness that some anatomical boundaries and tissue categories may be difficult to define reliably. Independent adjudication and assessment of interobserver variability may be valuable in ambiguous cases.

These considerations extend beyond maximising Dice scores.

For eventual clinical applications, a useful segmentation system would need to demonstrate reliable anatomical localisation, acceptable error characteristics, robustness across imaging conditions and appropriate handling of uncertainty.

The present model has not met those requirements and should not be used for clinical decision-making.

### 11.6 Overall Clinical Interpretation

The principal contribution of this project is not the development of a clinically deployable glioma segmentation model, but the demonstration of a reproducible approach to evaluating deep-learning predictions through both quantitative analysis and clinical anatomical reasoning.

The independent test results showed that relative-confidence post-processing improved average reference-based resection-cavity segmentation performance, with a positive subject-clustered bootstrap confidence interval for the mean paired Dice improvement.

However, substantial false-positive predictions persisted, individual examinations sometimes deteriorated markedly, and at least one examination demonstrated a clinically important disagreement between the model's predictions and the reference annotation.

Taken together, these findings reinforce the central principle of this project:

**Segmentation accuracy, anatomical plausibility and clinical validity are related but distinct concepts.**

A robust medical-imaging research workflow should evaluate all three rather than relying exclusively on aggregate segmentation metrics.

---

## 12. Limitations

Although the project demonstrates a reproducible workflow for postoperative glioma MRI segmentation, several methodological, computational and clinical limitations must be considered when interpreting the findings.

### 12.1 Limited Training and Validation Cohorts

The dataset contained 1,350 MRI examinations from 613 subjects, partitioned at the subject level into training, validation and held-out test cohorts.

However, the completed model-development experiments used only **180 training examinations and 24 validation examinations**, rather than the entire allocated training and validation partitions.

This restricted the diversity of imaging appearances encountered during training and reduced the precision with which validation performance could be estimated.

The small validation subset also increased the possibility that checkpoint selection and post-processing decisions were influenced by examination-specific characteristics.

Although independent evaluation was subsequently completed on all 198 held-out examinations, the limited development cohort remains an important constraint.

### 12.2 Single-Dataset Evaluation

The final model was evaluated on an independent held-out cohort comprising **198 examinations from 92 subjects**, with no subject overlap between the training, validation and test partitions.

This provides a stronger assessment of internal generalisation than training or validation performance alone.

However, all examinations originated from the same benchmark dataset.

The evaluation therefore does not establish generalisability to other institutions, MRI acquisition protocols, scanner manufacturers or patient populations.

External validation would be required before drawing broader conclusions about model robustness.

### 12.3 Incomplete Anatomical Representation

The model used a conventional 3D U-Net trained on 96 × 96 × 96 voxel patches.

Although patch-based training reduced computational requirements, the limited spatial context available within individual patches may have contributed to difficulties distinguishing anatomically similar structures.

Clinical review identified false-positive resection-cavity predictions involving normal CSF-containing structures, including the Sylvian fissure and probable ventricular spaces.

These findings are consistent with insufficient anatomical discrimination, although the contribution of patch size and receptive field was not independently tested.

The model also lacked explicit anatomical constraints enforcing the relationships between its segmentation outputs.

### 12.4 Independent Sigmoid Outputs and Anatomical Inconsistency

The model predicted whole tumour, tumour core, enhancing tumour and resection cavity using four independent sigmoid channels.

This representation permitted overlapping predictions without explicitly enforcing the nested relationships between tumour-associated regions or the mutually exclusive nature of reference enhancing-tumour and resection-cavity annotations.

Relative-confidence post-processing reduced one form of ET–RC conflict but did not address all possible anatomical inconsistencies.

In particular, false-positive cavity predictions involving normal CSF spaces could persist without simultaneous enhancing-tumour predictions.

Future architectures or loss functions incorporating appropriate anatomical relationships may improve segmentation consistency, but such approaches were not evaluated in this project.

### 12.5 Limitations of Relative-Confidence Post-Processing

Relative-confidence correction improved mean held-out resection-cavity Dice from 0.5221 to 0.5638.

However, the intervention was not uniformly beneficial.

Among 157 cavity-present examinations, RC Dice improved in 111, deteriorated in 22 and remained unchanged in 24.

The mean paired improvement of +0.0417 was substantially greater than the median improvement of +0.0015, indicating a heterogeneous distribution of effects.

Furthermore, the correction removed true-positive as well as false-positive cavity voxels, producing a reduction in mean RC recall.

Although false-positive cavity volume in cavity-absent examinations decreased by 70.2%, all 41 cavity-absent examinations retained some false-positive RC predictions.

The correction therefore represents a limited improvement in reference-based segmentation performance rather than a comprehensive solution to cavity misclassification.

### 12.6 Reference-Annotation Uncertainty

Postoperative MRI may demonstrate overlapping appearances of resection cavities, residual tumour, blood products, surgically altered tissue and reactive enhancement.

These features can complicate the interpretation of segmentation boundaries and tissue categories.

In examination `BraTS-GLI-02119-101`, the model assigned high enhancing-tumour probabilities to a region predominantly annotated as resection cavity.

This disagreement contributed to a substantial deterioration in RC Dice following relative-confidence correction.

Although multisequence anatomical review raised the possibility of annotation ambiguity, the available evidence was insufficient to establish an annotation error or determine the underlying tissue identity.

No independent expert adjudication or formal interobserver agreement analysis was performed.

Consequently, the reference annotations were retained as provided, and the findings were interpreted as potential annotation uncertainty rather than confirmed labelling error.

### 12.7 Limited Clinical Evaluation

The project primarily evaluated segmentation overlap using Dice, precision, recall and false-positive voxel counts.

These metrics provide useful measures of agreement with reference annotations but do not fully describe clinical significance.

For example, a small segmentation error near an eloquent structure may be clinically more consequential than a larger error elsewhere.

The study did not systematically evaluate surface-distance metrics, lesion-level detection, anatomical risk, volumetric measurement reliability or effects on clinical decision-making.

Qualitative anatomical review was performed on selected examinations but was not a blinded, systematic assessment by multiple independent specialists.

No prospective clinical evaluation was conducted.

### 12.8 Statistical and Methodological Constraints

The subject-clustered bootstrap analysis accounted for repeated examinations within subjects when estimating uncertainty in the mean paired RC Dice improvement.

The resulting 95% confidence interval was +0.0227 to +0.0625.

However, this interval describes uncertainty in the average effect within the evaluated dataset and does not establish that the correction improves every examination or provides clinical benefit.

The bootstrap analysis also does not address differences between institutions or other sources of dataset shift.

Additionally, detailed investigation of the most severely deteriorating examination was performed after held-out test results were available.

This review was appropriate for exploratory error analysis, but its findings should not be treated as independently validated evidence of annotation error or used to retrospectively modify the frozen test protocol.

### 12.9 Computational and Experimental Scope

The experiments were conducted using a consumer-grade workstation equipped with an NVIDIA RTX 4080 SUPER GPU with 16 GB of VRAM and 32 GB of system RAM.

The training subset, patch dimensions, batch configuration and model architecture were selected within these practical computational constraints.

The project did not systematically compare larger training cohorts, alternative architectures, multiple random seeds or extensive hyperparameter configurations.

Although a cavity-aware sampling experiment was performed, the investigation was not a comprehensive optimisation study.

Accordingly, the final results should be interpreted as the performance of a defined experimental pipeline rather than the maximum achievable performance on the dataset.

### 12.10 Overall Limitations

The principal limitations of this study are the restricted model-development cohort, evaluation within a single benchmark dataset, incomplete anatomical consistency, persistent false-positive cavity predictions and uncertainty in selected postoperative reference annotations.

The independent held-out evaluation strengthens the credibility of the reported performance, but it does not eliminate these limitations.

The results should therefore be regarded as an internally evaluated research demonstration rather than evidence of clinical readiness.

---

## 13. Future Experiments

The completed experiments established a baseline for postoperative glioma MRI segmentation and identified several limitations that could guide subsequent research.

In particular, the independent held-out evaluation demonstrated persistent false-positive resection-cavity predictions, heterogeneous responses to relative-confidence post-processing and potential uncertainty in selected postoperative reference annotations.

The following experiments are proposed as extensions of the current work. They have not been conducted and are not included in the reported results.

### 13.1 Expanding the Training Cohort

The current baseline model was trained using 180 MRI examinations, representing a relatively small proportion of the available training partition.

A direct next experiment would be to train the same 3D U-Net architecture using progressively larger subject-disjoint training subsets.

Potential comparisons could include:

- The existing 180-examination baseline.
- An intermediate training subset of approximately 400–500 examinations.
- The full allocated training partition of 936 examinations.

The model architecture, preprocessing, loss function and evaluation methodology should initially remain unchanged to isolate the effect of training-set size.

Performance could be compared using regional Dice, precision, recall, false-positive burden and examination-level error distributions.

Particular attention should be given to whether additional training diversity improves discrimination between normal CSF spaces and postoperative resection cavities.

This experiment would help determine whether the present limitations primarily reflect insufficient training diversity or more fundamental architectural and representational constraints.

### 13.2 Anatomically Informed Hard-Negative Sampling

Clinical review identified false-positive resection-cavity predictions involving the Sylvian fissure, probable ventricular spaces and heterogeneous tumour-associated abnormalities.

These findings suggest that random or cavity-focused patch sampling alone may not adequately expose the model to difficult non-cavity examples.

A future experiment could investigate **hard-negative sampling**, in which training patches are deliberately enriched for structures or regions previously associated with false-positive cavity predictions.

Potential targets include:

1. Normal CSF-containing structures, particularly the ventricular system and Sylvian fissures.
2. Tumour-associated regions with heterogeneous signal characteristics.
3. Postoperative abnormalities that resemble cavities but are assigned alternative reference labels.

A controlled comparison should distinguish the effects of hard-negative sampling from those of simply increasing the number of training patches.

The primary hypothesis would be that greater exposure to anatomically challenging negative examples reduces false-positive cavity predictions without substantially reducing sensitivity for genuine annotated cavities.

### 13.3 Anatomically Consistent Segmentation Outputs

The existing model predicts WT, TC, ET and RC using independent sigmoid channels.

Although this representation permits overlapping tumour-associated regions, it does not explicitly enforce their hierarchical relationships or the mutual exclusivity of ET and RC in the reference annotations.

A future experiment could compare the current independent-channel approach against an anatomically constrained output representation.

Possible approaches include:

- Predicting mutually exclusive underlying tissue classes and deriving WT, TC, ET and RC through deterministic combinations.
- Introducing differentiable consistency penalties during training.
- Applying constrained decoding to enforce predefined relationships between segmentation outputs.

The experimental design should distinguish improvements in anatomical consistency from improvements in reference-based segmentation accuracy.

A reduction in contradictory predictions would not, by itself, establish that the resulting masks are anatomically correct.

The impact on RC sensitivity, ET sensitivity and clinically important boundary errors should therefore be assessed separately.

### 13.4 Alternative Architectures and Greater Anatomical Context

The baseline model used a conventional 3D U-Net [6] trained on 96 × 96 × 96 voxel patches.

This configuration provided a practical balance between three-dimensional anatomical information and available GPU memory.

However, local image patches may not provide sufficient context to distinguish postoperative cavities from normal CSF-containing structures.

Future experiments could investigate architectures or inference strategies with greater anatomical context, including:

- Larger training patches where computationally feasible.
- Multiscale encoder–decoder architectures.
- Attention-based or transformer-assisted segmentation models.
- Coarse-to-fine approaches incorporating whole-brain anatomical information.

Potential candidate architectures include established medical-imaging segmentation frameworks such as nnU-Net and transformer-based U-Net variants.

Comparisons should use consistent subject-level data partitions and evaluation protocols.

The primary question would be whether increased anatomical context improves segmentation of postoperative cavities and reduces errors involving normal CSF spaces.

### 13.5 Improved Resection-Cavity Training Objectives

The current model was trained using a combined Dice and binary cross-entropy objective.

Although this approach supports region-overlap optimisation and voxel-level classification, it does not explicitly distinguish different anatomical sources of false-positive cavity predictions.

Future work could compare the baseline objective with alternative loss formulations, including:

- Region-weighted Dice and cross-entropy losses.
- Focal loss or focal Tversky loss.
- Boundary-aware segmentation losses.
- Anatomical consistency penalties.

These experiments should be designed to avoid simply increasing cavity recall at the expense of substantial false-positive predictions.

The previous cavity-aware sampling experiment demonstrated that greater emphasis on cavity-containing regions does not necessarily improve overall RC segmentation performance.

Consequently, future loss-function experiments should report precision, recall, false-positive burden and examination-level outcomes alongside Dice.

### 13.6 Systematic Investigation of Annotation Ambiguity

The held-out examination `BraTS-GLI-02119-101` demonstrated a substantial disagreement between the model's enhancing-tumour predictions and a region predominantly annotated as resection cavity.

Although post hoc multisequence review identified heterogeneous enhancement, the available evidence was insufficient to establish an annotation error or determine whether the enhancement represented viable tumour.

A future study could investigate such disagreements through systematic expert review.

A possible protocol would involve:

1. Identifying examinations with substantial ET–RC disagreement using predefined criteria.
2. Obtaining independent, blinded assessments from multiple clinicians experienced in postoperative neuro-oncological imaging.
3. Comparing expert interpretations with the supplied reference masks and model predictions.
4. Assessing interobserver agreement and the frequency of ambiguous annotations.
5. Incorporating longitudinal imaging or additional clinical information where available and appropriate.

The objective would not be to retrospectively alter the existing test results, but to investigate the reliability and clinical interpretation of postoperative segmentation labels.

Any revised annotations used for subsequent model development would require a separate evaluation protocol.

### 13.7 Expanded Quantitative Evaluation

The present study primarily used Dice, precision, recall and false-positive voxel counts.

Future evaluation could incorporate additional metrics that better characterise clinically relevant segmentation errors.

Potential measures include:

| Metric | Potential contribution |
|---|---|
| Hausdorff distance, including HD95 | Characterises boundary disagreement and large spatial errors |
| Average surface distance | Measures typical separation between predicted and reference boundaries |
| Absolute and relative volume error | Assesses accuracy of regional volume estimates |
| Lesion-level detection metrics | Distinguishes missed regions from fragmented or spurious predictions |
| Probability calibration metrics | Evaluates the reliability of model confidence estimates |
| Anatomical error localisation | Identifies errors involving specific intracranial structures |

These measures would complement, rather than replace, conventional segmentation overlap metrics.

Special handling would be required for absent regions, empty predictions and examinations containing multiple disconnected components.

### 13.8 External Validation and Robustness Assessment

The independent held-out test cohort provided an internal assessment of the frozen model, but all examinations originated from the same benchmark dataset.

A subsequent study should evaluate the model on independently sourced postoperative glioma MRI examinations, subject to appropriate data access and governance requirements.

External evaluation should investigate robustness across institutions, scanner characteristics, acquisition protocols and postoperative imaging appearances.

The original model and preprocessing protocol should be evaluated without modification before considering dataset-specific adaptation.

If model development is subsequently informed by the existing held-out test results, a new independent evaluation cohort would be required to assess the revised system without test-set contamination.

### 13.9 Prioritisation of Future Experiments

The proposed experiments differ in complexity, computational requirements and expected scientific value.

A practical sequence would be:

| Priority | Proposed experiment | Principal research question |
|---|---|---|
| 1 | Larger training cohort | Does greater training diversity improve generalisation and RC performance? |
| 2 | Hard-negative sampling | Can targeted training reduce normal CSF-space and tumour-associated false positives? |
| 3 | Anatomically consistent outputs | Can contradictory ET–RC predictions be reduced without sacrificing segmentation accuracy? |
| 4 | Systematic annotation review | How frequently do postoperative reference labels contain clinically ambiguous regions? |
| 5 | Alternative architectures and losses | Do more sophisticated models provide meaningful improvements over the baseline? |
| 6 | External validation | Do the findings generalise beyond the original benchmark dataset? |

This ordering is intended as a research planning framework rather than a claim that any particular intervention will necessarily improve performance.

The completed project provides the baseline results, error taxonomy and evaluation methodology required to investigate these questions systematically.

**None of these additional experiments is required to establish the findings reported in the present study.**

---

## 14. Longer-Term Research Directions

The present project establishes a foundational workflow for multimodal glioma MRI preprocessing, three-dimensional segmentation, quantitative evaluation and clinically informed error analysis.

Although the current model is not suitable for clinical deployment, the underlying technical framework could support several longer-term research directions relevant to neurosurgical planning, postoperative assessment and computational neuro-oncology.

The following applications are conceptual extensions of the completed work. None has been implemented or clinically validated as part of the present study.

### 14.1 Patient-Specific Three-Dimensional Reconstruction and Printing

Three-dimensional segmentation provides a potential starting point for generating patient-specific anatomical models.

The existing workflow produces volumetric masks representing whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC).

These masks could be converted into three-dimensional surface representations using established medical-image processing techniques.

A possible workflow would involve:

1. Loading the original MRI volumes and segmentation masks into a medical-imaging platform such as **3D Slicer**[8].
2. Reviewing and, where necessary, manually correcting the segmentation boundaries.
3. Generating three-dimensional surface meshes from the selected anatomical regions.
4. Performing mesh inspection, smoothing and repair.
5. Exporting suitable models in formats such as STL or 3MF for additive manufacturing.
6. Producing physical anatomical models using an appropriate 3D printing process.

Potential applications include anatomical education, preoperative case discussion and demonstration of spatial relationships between tumour-associated structures.

However, segmentation masks alone do not provide a complete representation of the operative anatomy.

A clinically informative model may also require integration of cortical surfaces, ventricular anatomy, vascular structures and other relevant anatomical landmarks.

Furthermore, the accuracy of a printed model is limited by the accuracy of its source imaging, segmentation and subsequent mesh processing.

**The present segmentation outputs would therefore require anatomical verification and correction before being considered for any clinical planning application.**

A feasible initial extension would be to demonstrate the technical pipeline using a de-identified research examination, without claiming operative planning accuracy.

### 14.2 Longitudinal Postoperative Imaging and Recurrence Prediction

A second research direction involves extending segmentation from individual MRI examinations to longitudinal postoperative assessment.

The present model evaluates each examination independently. It does not explicitly model changes in tumour-associated regions over time.

In clinical neuro-oncology, longitudinal imaging is central to evaluating postoperative evolution, treatment response and suspected disease progression.

A future workflow could combine:

- Longitudinal registration of MRI examinations.
- Segmentation of tumour-associated regions and postoperative cavities.
- Quantification of changes in regional volumes and spatial distribution.
- Extraction of imaging features associated with subsequent progression or recurrence.
- Statistical or machine-learning models for longitudinal outcome prediction.

For example, changes in enhancing-tumour volume, FLAIR abnormalities or the spatial distribution of new enhancement could be investigated as candidate predictors.

However, postoperative imaging changes are not specific for recurrent tumour.

Treatment-related effects, postoperative enhancement, radiation-associated changes and other non-neoplastic processes may resemble progression on conventional MRI.

Consequently, a recurrence-prediction study would require reliable longitudinal clinical labels, appropriate outcome definitions and consideration of treatment history.

The distinction between **detecting imaging change** and **predicting biologically meaningful recurrence** would be fundamental to the study design.

Potential outcomes could include radiologically adjudicated progression, time to progression or other clinically defined endpoints, depending on the availability and reliability of the underlying data.

The current BraTS segmentation labels alone are insufficient to establish such outcomes.

### 14.3 Integration With White-Matter Tract Anatomy

A particularly relevant longer-term direction is the integration of tumour segmentation with white-matter tractography.

Glioma surgery requires consideration not only of tumour location and extent but also of the relationship between pathological tissue and functionally important neural pathways.

Diffusion MRI and tractography can provide estimates of white-matter pathway trajectories, although these estimates are subject to technical and biological limitations [9].

A future research framework could integrate:

1. Structural MRI segmentation of tumour-associated regions.
2. Diffusion MRI preprocessing and fibre-orientation estimation.
3. Tractography reconstruction of selected white-matter pathways.
4. Spatial registration between structural and diffusion imaging.
5. Quantitative assessment of tumour–tract relationships.
6. Three-dimensional visualisation of the combined anatomical model.

Potential measurements could include estimated tumour-to-tract distance, spatial overlap and tract displacement.

Such analyses may be relevant to studying tumour-associated anatomical distortion and investigating imaging features associated with postoperative neurological outcomes.

However, tractography does not directly establish functional integrity, and reconstructed streamlines should not be interpreted as exact representations of individual axons.

Furthermore, reliable assessment of surgical risk would require additional information, potentially including functional imaging, neurophysiological mapping, operative findings and postoperative clinical outcomes.

The present project does not include diffusion MRI or tractography analysis.

This direction would therefore require a separate multimodal dataset and substantial additional methodological development.

### 14.4 Multimodal Modelling of Tumour Recurrence and Functional Risk

The preceding research directions could eventually be combined into a broader computational neuro-oncology framework.

Such a framework might integrate structural tumour segmentation, longitudinal imaging, diffusion-derived anatomical information and clinical variables.

Potential research questions include:

- Whether the spatial relationship between tumour-associated abnormalities and white-matter pathways is associated with patterns of subsequent progression.
- Whether longitudinal changes in segmented tumour regions improve prediction beyond single-time-point imaging features.
- Whether multimodal anatomical features are associated with postoperative neurological outcomes.
- Whether patient-specific three-dimensional visualisation improves interpretation of complex imaging findings.

These questions would require carefully defined clinical endpoints and sufficiently large, appropriately governed datasets.

They would also require evaluation strategies designed to avoid data leakage between repeated examinations from the same patient.

Any predictive model should undergo independent validation, calibration assessment and evaluation of clinical utility before consideration for clinical use.

### 14.5 Technical and Clinical Prerequisites

Progression from segmentation research to clinically meaningful applications would require additional capabilities beyond those demonstrated in the current project.

| Research direction | Key additional requirements |
|---|---|
| 3D reconstruction and printing | Anatomical verification, surface-mesh generation, mesh repair and dimensional accuracy assessment |
| Longitudinal imaging analysis | Serial MRI, image registration, consistent time-point definitions and reliable progression labels |
| Recurrence prediction | Longitudinal outcomes, treatment information, appropriate predictive modelling and external validation |
| Tract-aware modelling | Diffusion MRI, tractography, multimodal registration and clinically informed anatomical assessment |
| Functional-risk prediction | Relevant clinical outcomes, functional localisation data and prospective or external validation |

These prerequisites are important because technical feasibility does not necessarily imply clinical validity.

For example, producing a visually convincing three-dimensional tumour model does not establish anatomical accuracy, just as generating a recurrence-risk score does not establish predictive reliability.

Each proposed application would require its own research question, evaluation framework and validation strategy.

### 14.6 Longer-Term Research Vision

The broader objective of these potential extensions is to investigate how computational methods can integrate anatomical imaging, temporal disease behaviour and clinically relevant information in neurosurgical research.

The present project contributes a foundation in medical-image preprocessing, three-dimensional deep learning, segmentation evaluation, reproducible experimentation and clinically informed error analysis.

Its most important methodological lesson is that computational outputs must be interpreted within the anatomical and clinical context from which the data originate.

Future work could build on this foundation through patient-specific visualisation, longitudinal neuro-oncological analysis and multimodal modelling.

However, these remain distinct research objectives rather than demonstrated capabilities of the current segmentation system.

The immediate priority is to consolidate the completed segmentation experiments into a reproducible, clearly documented research portfolio before pursuing these extensions.

---

## 15. Reproducibility and Repository Organisation

### 15.1 Reproducibility Objectives

The project was developed with an emphasis on transparent methodology, traceable experimental decisions and reproducible evaluation.

The public GitHub repository is intended to document the complete research workflow, from initial MRI exploration and preprocessing through model development, clinical error analysis and independent held-out testing.

Three complementary levels of reproducibility are considered:

- **Methodological reproducibility:** The repository documents the model architecture, preprocessing procedures, training configuration, dataset partitioning strategy and evaluation methodology.
- **Analytical reproducibility:** Saved evaluation results can be used to reproduce reported summary statistics, comparisons and figures without repeating model inference.
- **Computational reproducibility:** Given the appropriate imaging data, software environment and trained checkpoint, the inference and evaluation procedures can be executed again.

These levels should be distinguished because the availability of source code does not necessarily imply that all data and trained model artefacts are publicly accessible.

### 15.2 Repository Structure

The project is maintained in the public GitHub repository:

[glioma-mri-segmentation](https://github.com/T-Y-Ang/glioma-mri-segmentation)

The repository is organised to separate exploratory analysis, reusable Python modules, experiment outputs and research documentation.

The principal project components are:

| Component | Purpose |
|---|---|
| `README.md` | Project overview, navigation, principal findings and instructions |
| `notebooks/01_mri_data_exploration.ipynb` | Dataset inspection, MRI visualisation and initial exploration |
| `notebooks/03_unet_model_development.ipynb` | Model development, training experiments, validation and clinical error analysis |
| `notebooks/04_model_evaluation.ipynb` | Completed independent held-out evaluation, regional performance analysis, paired resection-cavity comparisons,subject-clustered bootstrap analysis, anatomical failure-case investigation and reproducibility checks. |
| `src/preprocessing.py` | Reusable MRI preprocessing and label transformation functions |
| `src/losses.py` | Segmentation loss functions used during model development |
| `reports/glioma_segmentation_report.md` | Detailed research methodology, results, interpretation and limitations |
| `outputs/figures/` | Selected figures illustrating segmentation performance and anatomical findings |
| `data/` | Local data organisation, including raw, intermediate and processed files |
| `models/` | Model-related directory within the project structure |

The evaluation notebook is intended to provide a concise, independently executable record of the final assessment, rather than requiring readers to navigate the full experimental development notebook.

The repository structure may evolve as reusable evaluation functions and final figures are consolidated.

### 15.3 Data Provenance and Subject-Level Partitioning

The project uses the BraTS 2024 GLI dataset, comprising multimodal brain MRI examinations and associated segmentation annotations.

The dataset contains 1,350 examinations from 613 subjects.

The examinations were partitioned at the subject level into:

| Partition | Examinations | Subjects |
|---|---:|---:|
| Training | 936 | 429 |
| Validation | 216 | 92 |
| Held-out test | 198 | 92 |
| **Total** | **1,350** | **613** |

Subject-level partitioning was used to prevent examinations from the same subject appearing in different dataset partitions.

The completed model-development experiments used 180 training examinations and 24 validation examinations. Independent final evaluation was conducted on all 198 held-out test examinations.

The partition definitions were recorded in a local split manifest.

Because the manifest contains machine-specific absolute paths, the original file is not intended for direct public distribution without modification.

A portable version containing non-sensitive subject or examination identifiers and relative paths could support reproducibility, subject to the applicable dataset-sharing requirements.

### 15.4 Model and Training Configuration

The baseline segmentation model was a three-dimensional U-Net implemented using MONAI and PyTorch.

The principal configuration was:

| Parameter | Configuration |
|---|---|
| Architecture | 3D U-Net |
| Input channels | 4 MRI modalities |
| Output channels | WT, TC, ET and RC |
| Output activation | Independent sigmoid probabilities |
| Training patch size | 96 × 96 × 96 voxels |
| Optimiser | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.00001 |
| Training duration | 40 epochs |
| Training objective | Combined Dice and binary cross-entropy loss |
| Selected checkpoint | Baseline epoch 38 |

The baseline model contained approximately 4.81 million trainable parameters.

A subsequent cavity-aware sampling experiment was also conducted, but the original baseline checkpoint was retained for final evaluation because the alternative experiment did not demonstrate a convincing overall advantage.

The selected checkpoint and its experimental provenance were recorded before independent held-out testing.

### 15.5 Frozen Independent Evaluation Protocol

The independent held-out evaluation was performed using a predefined protocol established before examining the test results.

The protocol specified:

1. The selected baseline checkpoint from epoch 38.
2. Full-volume sliding-window inference using a 96 × 96 × 96 voxel region of interest, 25% overlap and a sliding-window batch size of one.
3. A primary evaluation using raw sigmoid predictions thresholded at 0.5.
4. A secondary evaluation using the validation-informed relative-confidence resection-cavity correction.
5. Regional Dice, precision, recall and false-positive measurements.
6. Paired examination-level comparisons between the raw and corrected predictions.

The relative-confidence correction modified only the RC channel.

A voxel was retained as RC when ET was not simultaneously predicted or when the RC probability was greater than or equal to the ET probability.

No model retraining, threshold optimisation or post-processing modification was performed using the held-out test results.

The frozen protocol was saved separately from the evaluation outputs to preserve the distinction between decisions made before testing and findings obtained afterward.

### 15.6 Saved Evaluation Results

The completed held-out evaluation generated a case-level metrics table containing **1,584 rows**, corresponding to:

- 198 MRI examinations.
- Two prediction methods: raw baseline and relative-confidence correction.
- Four segmentation regions per method.

The saved metrics include the information required to reproduce regional performance summaries and paired RC comparisons.

This permits the final results to be analysed without repeating GPU inference for every examination.

The principal evaluation outputs include:

| Output | Purpose |
|---|---|
| Frozen evaluation protocol | Documents model selection and evaluation decisions made before testing |
| Case-level metrics CSV | Supports independent recalculation of summary statistics |
| Regional performance summaries | Reports Dice, precision and recall for reference-present regions |
| Region-absent false-positive summaries | Characterises erroneous predictions when reference regions are absent |
| Paired RC analysis | Quantifies examination-level changes following correction |
| Subject-clustered bootstrap analysis | Estimates uncertainty in the mean paired RC Dice improvement |
| Clinical error-analysis figures | Illustrates selected anatomically important prediction errors |

Where practical and permitted, derived evaluation summaries and figures should be included in the public repository.

### 15.7 Independent Evaluation Notebook

The completed evaluation notebook, `notebooks/04_model_evaluation.ipynb`, consolidates the independent held-out test results, quantitative comparisons, statistical uncertainty analysis and clinically informed investigation of segmentation errors.

The notebook was developed to provide a transparent record of the final evaluation without requiring readers to navigate the full model-development history.

Its principal components include:

1. **Evaluation configuration and provenance:** Documentation of the selected epoch-38 checkpoint, preprocessing pipeline, inference settings and frozen evaluation protocol.
2. **Held-out test verification:** Loading and checking the saved case-level metrics for 198 examinations from 92 subjects.
3. **Regional segmentation performance:** Analysis of Dice, precision and recall across WT, TC, ET and RC, with appropriate handling of reference-absent regions.
4. **Resection-cavity post-processing comparison:** Paired analysis of raw and relative-confidence-corrected RC predictions, including examination-level improvements and deteriorations.
5. **False-positive burden:** Quantification of erroneous RC predictions in examinations without annotated resection cavities.
6. **Statistical uncertainty:** Subject-clustered bootstrap estimation of the confidence interval for the mean paired RC Dice improvement.
7. **Clinical failure-case investigation:** Reconstruction and anatomical analysis of the examination demonstrating the largest deterioration following correction, including ET and RC model-score comparisons.
8. **Visualisation:** Generation of eight figures illustrating aggregate performance, false-positive burden, post-processing trade-offs and selected anatomical errors.
9. **Final reproducibility checks:** Verification of saved experiment artifacts, test cohort coverage, metric-table completeness, figure availability and reference-absent metric handling.

The saved evaluation table contains 1,584 rows, representing two prediction methods and four anatomical regions for each of the 198 held-out examinations. These results allow the principal aggregate statistics and paired comparisons to be recalculated without repeating full-volume model inference.

The notebook also includes a targeted reconstruction of examination `BraTS-GLI-02119-101` using the frozen checkpoint and recorded preprocessing and inference settings. Reconstructed RC voxel counts and Dice scores matched the previously saved evaluation results, providing an additional reproducibility check for this selected failure case.

**Reproducibility boundaries:** Analyses based on saved metrics can be performed without rerunning GPU inference, provided the required derived files are accessible. Reconstructing individual MRI predictions requires the original imaging data, trained checkpoint and compatible software environment.

The final integrity-check cell is independently executable after a Jupyter kernel restart and verifies the availability and structural completeness of the principal local evaluation artifacts.

The notebook therefore provides a completed and independently reviewable record of the held-out evaluation, while recognising that full computational reproduction remains dependent on separately stored imaging data and model artifacts.

### 15.8 Software Environment and Computational Hardware

The experiments were conducted in a Python 3.11 environment using PyTorch, MONAI, NiBabel, NumPy and related scientific Python libraries.

The primary computing hardware comprised:

- NVIDIA RTX 4080 SUPER GPU with 16 GB of VRAM.
- 32 GB of system RAM.
- Windows with Ubuntu running through Windows Subsystem for Linux.

The model-development environment used CUDA-enabled PyTorch for GPU acceleration.

Exact package versions and relevant dependencies should be recorded in an environment specification to support future reproduction.

Hardware-dependent differences, library versions and nondeterministic GPU operations may affect exact numerical reproduction of training results.

Consequently, reproducibility should be assessed at the level of documented methodology and expected numerical agreement rather than assuming bit-for-bit identical retraining outcomes.

### 15.9 Data Governance and Public Repository Boundaries

The public repository is intended to distribute source code, documentation, selected figures and permissible derived results.

The original BraTS imaging dataset is maintained separately from the version-controlled repository.

Large experimental artefacts, including trained checkpoints and local inference outputs, may also be stored outside GitHub because of their size and machine-specific storage requirements.

Local filesystem paths, authentication credentials and dataset-access information should not be included in publicly committed files.

Any public distribution of derived outputs, subject identifiers or model artefacts should comply with the applicable dataset terms and governance requirements.

The repository should therefore distinguish clearly between:

- Publicly available code and documentation.
- Publicly shareable derived results.
- Externally obtained datasets requiring separate access.
- Locally stored checkpoints and experimental artefacts.

This separation supports transparency without implying unrestricted redistribution rights over the underlying imaging data.

### 15.10 Reproducibility Status

The principal experimental and analytical components of the postoperative glioma MRI segmentation project have been completed.

The documented workflow encompasses dataset exploration, multimodal MRI preprocessing, subject-level data partitioning, baseline 3D U-Net training, cavity-aware sampling experiments, validation-stage clinical error analysis, frozen evaluation protocol selection and independent held-out testing.

The final evaluation was conducted on 198 MRI examinations from 92 subjects. Its results were consolidated in `notebooks/04_model_evaluation.ipynb`, including regional performance comparisons, resection-cavity false-positive analysis, paired post-processing comparisons, subject-clustered bootstrap estimation and investigation of a selected anatomical failure case.

Eight evaluation figures were generated and saved to `outputs/figures/`.

#### Verification of Saved Experimental Artifacts

A final integrity check was performed after completion of the evaluation notebook.

The verification confirmed:

- Availability of the selected epoch-38 model checkpoint.
- Availability of the frozen evaluation protocol.
- Availability of the saved held-out test metrics.
- Presence of all eight evaluation figures.
- Complete coverage of 198 examinations from 92 subjects.
- A total of 1,584 evaluation rows, representing two prediction methods and four anatomical regions per examination.
- Absence of duplicate examination–method–region records.
- Appropriate exclusion of reference-absent regions from Dice calculations.

All final integrity checks passed.

Additionally, the examination demonstrating the largest deterioration in resection-cavity Dice following post-processing was reconstructed using the original preprocessing implementation, frozen model checkpoint and recorded inference parameters.

The reconstructed voxel counts and Dice scores matched the saved evaluation results, supporting the reproducibility of the selected failure-case analysis.

#### Levels of Reproducibility Achieved

**Methodological reproducibility:** The repository documents the experimental design, preprocessing methodology, model architecture, training configuration, evaluation protocol and interpretation of results.

**Analytical reproducibility:** The saved case-level metrics support recalculation of the principal regional performance summaries, paired comparisons and statistical uncertainty estimates without repeating full-volume MRI inference.

**Local computational reproducibility:** The original imaging data, selected checkpoint and experimental artifacts remain available in the local research environment, enabling the documented inference procedures to be repeated using compatible software and hardware.

These levels of reproducibility do not imply that the complete experiment can be reproduced from the public repository alone.

#### Public Reproducibility Limitations

The original BraTS imaging dataset, trained model checkpoint and certain experiment artifacts are stored separately from the version-controlled repository.

Consequently, readers cloning the public repository may inspect the source code, methodology, documentation and committed figures, but full inference reproduction requires additional data and artifacts.

Similarly, analytical reproduction from saved case-level metrics requires access to those metrics. Their local availability does not establish that they are included in the public repository.

Exact software dependencies, portable configuration files and publicly shareable derived results should be reviewed as part of the final repository audit.

#### Overall Status

The experimental evaluation, statistical analysis, clinical failure-case investigation, evaluation notebook and local artifact-integrity verification are complete.

The remaining work concerns public repository finalisation, including documentation consistency, dependency specification, navigation, appropriate sharing of derived results and verification of reproducibility instructions.

The project therefore supports a completed, internally evaluated research study with documented methodology and locally verified experimental artifacts.

**It does not yet constitute a fully self-contained, publicly reproducible release or a clinically validated segmentation system.**

---

## 16. Current Project Status

### 16.1 Completed Research Activities

The principal experimental objectives of this project have been completed.

The study progressed from multimodal MRI exploration and preprocessing to baseline 3D U-Net development, validation-stage anatomical error analysis, a cavity-aware sampling experiment and independent held-out evaluation.

The final evaluation was conducted using a predefined protocol, with the selected baseline checkpoint and post-processing methods established before examining the held-out test results.

| Project component | Status |
|---|---|
| BraTS 2024 GLI dataset acquisition and exploration | Completed |
| Multimodal MRI preprocessing and label conversion | Completed |
| Subject-level training, validation and test partitioning | Completed |
| Baseline 3D U-Net development and training | Completed |
| Validation performance assessment | Completed |
| Clinical review of selected segmentation errors | Completed |
| Cavity-aware sampling experiment | Completed |
| Validation-informed post-processing comparison | Completed |
| Final checkpoint selection and evaluation protocol | Completed |
| Independent held-out evaluation of 198 examinations | Completed |
| Paired resection-cavity performance analysis | Completed |
| Subject-clustered bootstrap analysis | Completed |
| Post hoc clinical review of a potential annotation disagreement | Completed |
| Documentation of experimental findings and limitations | Completed |

No additional model training or hyperparameter optimisation is required to complete the current experimental study.

### 16.2 Principal Findings

The selected baseline 3D U-Net achieved a macro-average Dice of **0.6525** across the four segmentation regions on the independent held-out test cohort.

Whole-tumour segmentation demonstrated the strongest performance, with a mean Dice of 0.8126, while resection-cavity segmentation remained the most challenging region, with a mean Dice of 0.5221.

Validation-informed relative-confidence post-processing increased mean RC Dice to **0.5638** and reduced false-positive cavity voxel burden by **70.2%** in examinations without an annotated resection cavity.

The mean paired RC Dice improvement was **+0.0417**, with a 95% subject-clustered bootstrap confidence interval of **+0.0227 to +0.0625**.

However, the correction did not eliminate false-positive cavity predictions and caused substantial deterioration in some individual examinations.

Clinical error analysis identified normal CSF-space misclassification, tumour-associated cavity false positives and potential ambiguity in selected postoperative reference annotations.

These findings demonstrate the importance of evaluating segmentation models using both quantitative performance measures and clinically informed anatomical interpretation.

### 16.3 Remaining Repository Finalisation

The principal experimental and analytical work has been completed, including independent held-out evaluation, statistical analysis, clinical failure-case investigation and verification of saved evaluation artifacts.

The standalone evaluation notebook, `notebooks/04_model_evaluation.ipynb`, has been completed and includes a final integrity-check cell confirming the availability and structural completeness of the principal local experiment artifacts.

Eight final evaluation figures have also been generated and saved.

The remaining work primarily concerns documentation consistency, public accessibility and reproducibility of the GitHub portfolio.

| Project component | Purpose | Current status |
|---|---|---|
| Independent held-out evaluation | Assess the frozen model on 198 examinations from 92 subjects | Completed |
| Standalone evaluation notebook | Document final metrics, statistical comparisons, visualisations and failure-case analysis | Completed |
| Final evaluation figures | Present regional performance, RC post-processing effects and anatomical errors | Completed — eight figures saved |
| Local evaluation integrity checks | Verify checkpoint, protocol, metrics, cohort coverage and figures | Completed — all checks passed |
| Research report consistency review | Ensure methods, results, interpretation and conclusions reflect completed experiments | In progress |
| Public evaluation summaries | Enable independent inspection and recalculation of permissible derived results | Pending verification and preparation |
| README and repository navigation | Present the project clearly to clinical and technical readers | Pending final review |
| Software environment specification | Document package versions and dependencies required for reproduction | Pending verification |
| Reproducibility and public-file audit | Check repository paths, notebook execution requirements, links and data-sharing boundaries | Pending |
| Final GitHub release | Publish a coherent, documented research portfolio | Pending |

The remaining tasks do not require additional model training, hyperparameter optimisation or repeated held-out test inference.

In particular, further modification of the existing model or relative-confidence correction in response to held-out test findings would constitute a new research iteration and require an appropriately independent evaluation strategy.

The immediate priority is therefore to ensure that the completed work can be understood, inspected and, where the necessary data and artifacts are available, reproduced by other researchers.

Once the remaining documentation and repository checks are complete, the project can be designated as a finalised portfolio release.

The current model remains a research demonstration and is not suitable for clinical deployment.

### 16.4 Scope of the Completed Study

The completed project establishes a reproducible workflow for:

- Processing multimodal postoperative glioma MRI.
- Training a three-dimensional segmentation model.
- Evaluating segmentation performance on a subject-disjoint held-out cohort.
- Comparing a predefined post-processing intervention against raw predictions.
- Quantifying examination-level effects and statistical uncertainty.
- Investigating anatomically meaningful segmentation errors.

The study does not establish clinical deployment readiness.

External validation, systematic expert annotation review, prospective evaluation and investigations of clinical utility remain outside the scope of the completed work.

Similarly, three-dimensional printing, longitudinal recurrence prediction and tract-aware modelling are longer-term research directions rather than implemented project components.

### 16.5 Completion Milestone

The experimental phase of the glioma MRI segmentation project is **complete**.

The remaining milestone is to consolidate the results into a clearly documented, independently reviewable and reproducible public GitHub portfolio.

Once the remaining repository documentation, software environment specification, public reproducibility checks and final GitHub quality audit are completed, the project can be designated as a finalised portfolio release.

Subsequent model improvements or expanded clinical applications should be treated as separate research iterations with their own objectives and evaluation protocols.

---

## 17. Final Conclusions

This project investigated the development and evaluation of a three-dimensional deep-learning framework for multimodal postoperative glioma MRI segmentation using the BraTS 2024 GLI dataset.

A conventional 3D U-Net was trained to predict four segmentation regions: whole tumour (WT), tumour core (TC), enhancing tumour (ET) and resection cavity (RC). The study incorporated subject-level dataset partitioning, quantitative validation, clinical anatomical error analysis, an alternative cavity-aware sampling experiment and independent held-out evaluation using a predefined protocol.

The selected baseline model achieved a macro-average Dice of **0.6525** across 198 held-out MRI examinations from 92 subjects. Whole-tumour segmentation demonstrated the strongest performance, while resection-cavity segmentation remained particularly challenging.

A validation-informed relative-confidence post-processing intervention improved mean RC Dice from **0.5221 to 0.5638** and reduced false-positive cavity voxel burden by **70.2%** in examinations without an annotated resection cavity.

The mean paired RC Dice improvement was **+0.0417**, with a 95% subject-clustered bootstrap confidence interval of **+0.0227 to +0.0625**.

However, the improvement was heterogeneous. Some examinations demonstrated substantial deterioration, and false-positive cavity predictions persisted despite correction.

Clinical anatomical review identified several important limitations of the segmentation system, including misclassification of normal CSF-containing structures, confusion between tumour-associated abnormalities and postoperative cavities, and potential ambiguity in selected reference annotations.

These findings demonstrate that improvements in numerical segmentation overlap do not necessarily establish anatomical correctness or clinical superiority.

The project therefore supports three principal conclusions.

**First**, a reproducible postoperative glioma segmentation workflow can be developed and evaluated using established deep-learning techniques and consumer-grade computing hardware.

**Second**, relatively simple, validation-informed post-processing can improve average reference-based segmentation performance, but its benefits must be assessed alongside individual failures, sensitivity losses and persistent false-positive predictions.

**Third**, clinically informed anatomical review provides an essential complement to conventional segmentation metrics by identifying error patterns and interpretive uncertainties that may otherwise remain unrecognised.

The study is limited by its relatively small model-development cohort, evaluation within a single benchmark dataset, incomplete anatomical consistency and absence of external or prospective clinical validation.

Accordingly, the results should be interpreted as a retrospective research demonstration rather than evidence of clinical deployment readiness.

Beyond its immediate segmentation findings, the project establishes a methodological foundation for future investigations into anatomically informed medical-image analysis, patient-specific three-dimensional reconstruction, longitudinal neuro-oncological imaging and multimodal computational neurosurgery.

**The central conclusion is that reliable medical-image segmentation requires more than numerical agreement with reference masks: it requires careful consideration of anatomical plausibility, reference uncertainty and the clinical significance of model errors.**

---

## 18. References

### 18.1 Dataset and Postoperative Glioma Segmentation

1. Correia de Verdier M, Saluja R, Gagnon L, et al. The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI. *arXiv*. 2024. arXiv:2405.18368.
   https://arxiv.org/abs/2405.18368

2. Baig S, Vidic I, Mastorakos GM, et al. Segmentation of pre- and posttreatment diffuse glioma tissue subregions including resection cavities. *Neurooncol Adv*. 2024;6(1):vdae140.
   https://doi.org/10.1093/noajnl/vdae140

### 18.2 Deep-Learning Architectures and Segmentation Methods

3. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In: *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*. Lecture Notes in Computer Science. Vol 9351. Springer; 2015:234–241.
   https://doi.org/10.1007/978-3-319-24574-4_28

4. Çiçek Ö, Abdulkadir A, Lienkamp SS, Brox T, Ronneberger O. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In: *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016*. Lecture Notes in Computer Science. Vol 9901. Springer; 2016:424–432.
   https://arxiv.org/abs/1606.06650

5. Milletari F, Navab N, Ahmadi SA. V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. In: *2016 Fourth International Conference on 3D Vision (3DV)*. IEEE; 2016:565–571.
   https://doi.org/10.1109/3DV.2016.79

6. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods*. 2021;18(2):203–211.
   https://doi.org/10.1038/s41592-020-01008-z

### 18.3 Medical-Imaging Software and Three-Dimensional Reconstruction

7. Cardoso MJ, Li W, Brown R, et al. MONAI: An open-source framework for deep learning in healthcare. *arXiv*. 2022. arXiv:2211.02701.
   https://arxiv.org/abs/2211.02701

8. Fedorov A, Beichel R, Kalpathy-Cramer J, et al. 3D Slicer as an image computing platform for the Quantitative Imaging Network. *Magn Reson Imaging*. 2012;30(9):1323–1341.
   https://doi.org/10.1016/j.mri.2012.05.001

### 18.4 Longitudinal Imaging and Future Neuro-Oncological Applications

9. Cozzi FM, Mayrand RC, Wan Y, Price SJ. Predicting glioblastoma progression using MR diffusion tensor imaging: A systematic review. *J Neuroimaging*. 2024;35(1):e13251. doi:[10.1111/jon.13251](https://doi.org/10.1111/jon.13251).

   PMCID: PMC11626419. PMID: 39648937.

### 18.5 Original Experimental Work

The model-development experiments, validation analyses, relative-confidence post-processing comparison, independent held-out test results, paired resection-cavity analysis and subject-clustered bootstrap estimates presented in this report constitute original analyses performed for the present project.

These findings are documented in the accompanying research report, notebooks and saved experimental outputs. They should not be attributed to the external publications listed above.
---

**Document status:** Completed experimental study — final repository preparation in progress.

The baseline model-development experiments, independent held-out evaluation, statistical analysis and clinical failure-case investigation have been completed. The research report documents the final findings, methodology, limitations and proposed future directions.

Remaining work is limited to repository documentation, public reproducibility checks and final portfolio presentation. Future model improvements or additional validation studies will be documented as separate research iterations.
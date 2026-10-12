
"""
BraTS 2024 GLI MRI Preprocessing Utilities
==========================================

Project:
    Glioma MRI Segmentation Using Deep Learning

Purpose:
    Provide reusable preprocessing functions for the BraTS 2024 glioma MRI segmentation project.

    These utilities standardise MRI inputs and segmentation labels so that training, validation and testing use consistent
    anatomical definitions and preprocessing operations.

MRI inputs:
    Four co-registered MRI sequences are expected, in this order:

        1. T1n - Native T1-weighted MRI
        2. T1c - Contrast-enhanced T1-weighted MRI
        3. T2w - T2-weighted MRI
        4. T2f - T2-FLAIR MRI

Segmentation outputs:
    Four binary segmentation channels are generated:

        WT - Whole tumour
        TC - Tumour core
        ET - Enhancing tissue
        RC - Resection cavity

    These channels are not mutually exclusive. For example, enhancing tissue is included within tumour core, which is itself included within whole tumour.

Preprocessing workflow:
    1. Load MRI volumes and segmentation annotations.
    2. Convert data to channel-first format.
    3. Standardise image orientation to RAS.
    4. Normalise MRI intensities independently by modality.
    5. Crop images to the nonzero MRI foreground.
    6. Convert images and labels to PyTorch tensors.
    7. Convert categorical annotations into binary region masks.

Notes:
    - The current pipeline is deterministic and intended for validation-style preprocessing.
    - Random patch sampling and data augmentation are not performed here.
    - Foreground cropping preserves positive and negative nonzero intensities after normalisation.
    - All four MRI modalities and the segmentation must be spatially aligned before preprocessing.
    - This module does not perform model training or inference.

Dependencies:
    PyTorch, MONAI

Related notebooks:
    notebooks/02_preprocessing_pipeline.ipynb
    notebooks/03_unet_model_development.ipynb
"""

# =============================================================
# 1. IMPORTS
# =============================================================

import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    EnsureTyped,
    MapTransform,
)


# =============================================================
# 2. BRATS SEGMENTATION REGION DEFINITIONS
# =============================================================

# BraTS 2024 post-treatment segmentation label convention:
#
# 0 = Background
# 1 = Non-enhancing tumour core
# 2 = Surrounding non-enhancing FLAIR hyperintensity
# 3 = Enhancing tissue
# 4 = Resection cavity
#
# The model predicts four binary regions rather than a
# single mutually exclusive categorical segmentation.
#
# WT includes labels 1, 2 and 3.
# TC includes labels 1 and 3.
# ET includes label 3.
# RC includes label 4.
#
# Dictionary insertion order determines the output channel order.

BRATS_REGIONS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
    "RC": (4,),
}


# =============================================================
# 3. FOREGROUND SELECTION
# =============================================================

def select_nonzero_foreground(image):
    """
    Identify voxels with nonzero MRI intensity.

    This function is used by MONAI's CropForegroundd transform
    to identify the spatial region containing MRI signal.

    Unlike positive-only selection (image > 0), this function
    retains negative intensities produced by normalisation.

    The channel dimension is preserved, allowing MONAI to
    determine the foreground across all MRI modalities.

    Parameters
    ----------
    image : torch.Tensor or numpy.ndarray
        Channel-first MRI image with shape (C, H, W, D).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Boolean mask with the same shape as the input.

    Notes
    -----
    This function identifies foreground voxels; CropForegroundd
    subsequently calculates a bounding box around them.
    """

    return image != 0


# =============================================================
# 4. SEGMENTATION LABEL CONVERSION
# =============================================================

class ConvertBraTSLabelsd(MapTransform):
    """
    Convert categorical BraTS labels into binary region masks.

    The original segmentation contains one categorical channel
    with integer labels.

    This transform converts it into four binary channels
    representing the regions defined in BRATS_REGIONS.

    Output channel order
    --------------------
    Channel 0 : WT - Whole tumour
    Channel 1 : TC - Tumour core
    Channel 2 : ET - Enhancing tissue
    Channel 3 : RC - Resection cavity

    Parameters
    ----------
    keys : sequence
        Dictionary keys containing segmentation annotations.

    Returns
    -------
    dict
        Updated dictionary containing four-channel segmentation
        tensors with shape (4, H, W, D) and dtype float32.

    Notes
    -----
    Binary channels are suitable for independent sigmoid
    predictions and overlapping-region segmentation losses.

    This transform should be applied after spatial operations
    on the original categorical segmentation labels.
    """

    def __call__(self, data):

        # Create a shallow copy to avoid replacing entries in
        # the input dictionary itself.
        result = dict(data)

        for key in self.key_iterator(result):

            # The original segmentation has shape (1, H, W, D).
            # Remove the singleton channel dimension.
            categorical_label = result[key][0]

            # Generate one binary mask for each anatomical region.
            # torch.isin() identifies voxels belonging to the
            # categorical labels assigned to that region.
            binary_regions = [
                torch.isin(
                    categorical_label,
                    torch.as_tensor(
                        region_values,
                        device=categorical_label.device
                    )
                )
                for region_values in BRATS_REGIONS.values()
            ]

            # Stack binary masks into four output channels.
            # Convert boolean masks to float32 for compatibility
            # with the segmentation loss function.
            result[key] = torch.stack(
                binary_regions,
                dim=0
            ).float()

        return result


# =============================================================
# 5. DETERMINISTIC VALIDATION PREPROCESSING
# =============================================================

def create_validation_transforms():
    """
    Construct the deterministic BraTS preprocessing pipeline.

    This function prepares MRI images and reference annotations
    for validation, evaluation and preprocessing verification.

    Transform sequence
    ------------------
    1. Load MRI volumes and segmentation labels.
    2. Ensure channel-first tensor layout.
    3. Reorient images and labels to RAS.
    4. Normalise nonzero MRI intensities per modality.
    5. Crop images and labels to the MRI foreground.
    6. Convert MRI images to float32 tensors.
    7. Convert categorical labels to int64 tensors.
    8. Generate four binary segmentation channels.

    Returns
    -------
    monai.transforms.Compose
        Deterministic preprocessing pipeline.

    Expected input
    --------------
    Dictionary with:

        "image": list of four MRI file paths
        "label": segmentation file path

    Expected output
    ---------------
    Dictionary containing:

        "image": float32 tensor, shape (4, H, W, D)
        "label": float32 tensor, shape (4, H, W, D)

    Notes
    -----
    Spatial dimensions depend on the foreground crop and are
    not necessarily divisible by the U-Net downsampling factor.

    Full-volume inference may therefore require additional
    spatial handling, such as sliding-window inference.
    """

    return Compose([

        # -----------------------------------------------------
        # STEP 1 — LOAD MRI AND SEGMENTATION
        # -----------------------------------------------------

        LoadImaged(
            keys=("image", "label")
        ),

        # -----------------------------------------------------
        # STEP 2 — STANDARDISE CHANNEL ORDER
        # -----------------------------------------------------

        EnsureChannelFirstd(
            keys=("image", "label")
        ),

        # -----------------------------------------------------
        # STEP 3 — STANDARDISE SPATIAL ORIENTATION
        # -----------------------------------------------------

        Orientationd(
            keys=("image", "label"),
            axcodes="RAS"
        ),

        # -----------------------------------------------------
        # STEP 4 — NORMALISE MRI INTENSITIES
        # -----------------------------------------------------
        #
        # Normalise each MRI modality independently.
        # Exclude zero-valued background voxels from the
        # intensity statistics.

        NormalizeIntensityd(
            keys="image",
            nonzero=True,
            channel_wise=True
        ),

        # -----------------------------------------------------
        # STEP 5 — CROP TO MRI FOREGROUND
        # -----------------------------------------------------
        #
        # Use nonzero intensities from the four MRI modalities.
        # Apply the same spatial crop to images and labels.

        CropForegroundd(
            keys=("image", "label"),
            source_key="image",
            select_fn=select_nonzero_foreground
        ),

        # -----------------------------------------------------
        # STEP 6 — CONVERT TO PYTORCH TENSORS
        # -----------------------------------------------------

        EnsureTyped(
            keys="image",
            dtype=torch.float32
        ),

        EnsureTyped(
            keys="label",
            dtype=torch.int64
        ),

        # -----------------------------------------------------
        # STEP 7 — CONVERT SEGMENTATION REGIONS
        # -----------------------------------------------------

        ConvertBraTSLabelsd(
            keys=("label",)
        ),
    ])

🧠 MRI-to-CT Translation with 3D U-Net + CBAM

This repository contains the implementation of an enhanced 3D U-Net model integrated with a Convolutional Block Attention Module (CBAM) for translating MRI scans into CT scans.
The CBAM module adaptively refines both spatial and channel-wise feature representations within the skip connections, enabling the network to focus on the most informative anatomical structures during synthesis.

The model was trained and evaluated on the SynthRAD2023 public dataset, demonstrating improved quantitative accuracy (e.g., lower MAE) and better structural consistency compared to the baseline 3D U-Net.

Links :
[SynthRAD2023 Dataset](https://zenodo.org/records/7260705)   
[![Hugging Face](https://img.shields.io/badge/Demo-HuggingFace-yellow.svg)](https://huggingface.co/spaces/jihane12/ct-mri_translation)

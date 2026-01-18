# EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.16442-b31b1b.svg)](https://arxiv.org/abs/2510.16442)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://11ouo1.github.io/edvd-llama/)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ER--FF%2B%2Bset-blue)](https://huggingface.co/datasets/Codebee/ER-FFppset)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

</div>

---

## 📢 News & Updates

- **[2025-01]**: We have released the **ER-FF++set** dataset on HuggingFace!
- **[Status]**: 🚧 **The code for EDVD-LLaMA is currently under preparation and will be released immediately upon the acceptance of the paper.** Please stay tuned!

---

## 📖 Abstract

This repository contains the official implementation of the paper: **"EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning"**.

Deepfake detection has traditionally focused on binary classification, often lacking transparency in decision-making. In this work, we propose **EDVD-LLaMA**, a novel framework that leverages Multimodal Large Language Models (MLLMs) to not only detect deepfakes but also provide verifiable, reasoning-based explanations.

To support this task, we introduce the **ER-FF++set**, a high-quality dataset constructed from **FaceForensics++**, enriched with structured visual-language annotations to train models for explainable forensics.

<div align="center">
  <img src="assets/framework.png" alt="EDVD-LLaMA Framework" width="800"/>
  <br>
  <em>(Figure: Overview of the EDVD-LLaMA Framework. Please upload your framework image to an 'assets' folder.)</em>
</div>

## 📂 ER-FF++set Dataset

We present **ER-FF++set** (Explainable Reasoning FF++ Benchmark Dataset), specifically designed for the Explainable Deepfake Video Detection (EDVD) task.

- **Source**: The video clips are derived exclusively from the **FaceForensics++** dataset, covering five mainstream manipulation techniques (Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTexture).
- **Annotations**: Unlike the original binary labels, our dataset includes:
    - **Forgery Masks**: Pixel-level ground truth.
    - **Structured Rationale**: Detailed textual explanations describing *why* a video is fake (e.g., "irregular blinking," "mouth artifacts").
    - **QA Pairs**: Instruction-tuning data for MLLMs.

You can download the dataset here: [🤗 HuggingFace Link](https://huggingface.co/datasets/Codebee/ER-FFppset)

## 🏆 Performance Benchmark

Our model achieves state-of-the-art performance on the ER-FF++set benchmark, significantly outperforming general-purpose Video-MLLMs.

| Model | Accuracy (%) | AUC (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: |
| Video-LLaVA | 51.63 | 54.12 | 51.94 |
| Video-ChatGPT | 52.27 | 56.53 | 52.38 |
| VideoLLaMA3 | 72.48 | 75.93 | 73.50 |
| **EDVD-LLaMA (Ours)** | **84.75** | **87.64** | **85.13** |


## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@misc{sun2025edvdllamaexplainabledeepfakevideo,
      title={EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning}, 
      author={Haoran Sun and Chen Cai and Huiping Zhuang and Kong Aik Lee and Lap-Pui Chau and Yi Wang},
      year={2025},
      eprint={2510.16442},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.16442}, 
}

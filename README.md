# MLADDC: Multi-Lingual Audio Deepfake Detection Corpus

[![Conference](https://img.shields.io/badge/ICASSP-2025-blue)](https://2025.ieeeicassp.org/)
[![License](https://img.shields.io/badge/License-Research%20Use-green)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Source](https://img.shields.io/badge/Source-VoxLingua107-purple)](http://bark.phon.ioc.ee/voxlingua107/)

> **Official repository** for the paper:  
> *"MLADDC: Multi-Lingual Audio Deepfake Detection Corpus"*  
> [Arth J. Shah](https://www.linkedin.com/in/arth-shah-18b4a0245), [Ravindrakumar M. Purohit](https://iamshreeji-copy1.github.io/ravindrakumar_purohit/index.html), [Dharmendra H. Vaghera](https://www.linkedin.com/in/dharmendra-vaghera-95677b256/), and [Hemant A. Patil](https://sites.google.com/site/hemantpatildaiict/)  
> [Speech Research Lab](https://sites.google.com/site/speechlabdaiict/home), Dhirubhai Ambani Institute of Information and Communication Technology (DA-IICT), Gandhinagar  
> *Submitted to ICASSP 2025, Hyderabad, India*

> ⚠️ **Note:** This paper is currently under review. The dataset and full codebase will be made publicly available upon acceptance.

---

## Overview

Audio deepfakes — synthetically generated or voice-converted speech — pose a growing threat to the integrity of spoken communication. Most existing audio deepfake detection benchmarks are built on English-only or limited-language corpora, which restricts the generalizability of detection models across the world's linguistic diversity.

**MLADDC** addresses this gap by introducing a large-scale, balanced, multilingual corpus for audio deepfake detection, sourced from the [VoxLingua107](http://bark.phon.ioc.ee/voxlingua107/) dataset. It spans **20 languages** — including multiple Indic languages — with carefully selected audio clips stratified by duration to ensure balanced representation across all language groups.

---

## Repository Contents

```
.
├── requirements.txt        # Python dependency specifications
└── README.md               # This file
```

> Full dataset, baseline models, and experiment notebooks will be released upon paper acceptance.

---

## Dataset Description

### Source

MLADDC is derived from [**VoxLingua107**](http://bark.phon.ioc.ee/voxlingua107/), a large-scale multilingual speech dataset sourced from YouTube. From the original corpus, audio samples were carefully selected per language with stratified sampling across four duration buckets (0–5s, 5–10s, 10–15s, 15–20s) to ensure balanced language-level representation.

### Languages Covered (20 Languages)

| Language         | Code | Selected Duration (hrs) |
|------------------|------|------------------------|
| Russian          | ru   | 11.39                  |
| French           | fr   | 11.38                  |
| Arabic           | ar   | 11.34                  |
| Spanish          | es   | 11.37                  |
| Vietnamese       | vi   | 11.28                  |
| Mandarin Chinese | zh   | 11.37                  |
| English          | en   | 11.32                  |
| Hindi            | hi   | 11.42                  |
| Portuguese       | pt   | 11.37                  |
| Sanskrit         | sa   | 8.92                   |
| Bahasa Indonesia | id   | 11.35                  |
| Bengali          | bn   | 11.36                  |
| Finnish          | fi   | 11.52                  |
| Japanese         | ja   | 11.34                  |
| Gujarati         | gu   | 11.36                  |
| Tamil            | ta   | 11.40                  |
| Punjabi          | pa   | 11.36                  |
| Urdu             | ur   | 11.35                  |
| Swedish          | sv   | 11.39                  |
| German           | de   | 11.54                  |

> Sanskrit (`sa`) has a smaller selected pool (8.92 hrs) due to limited availability in the original VoxLingua107 source.

### Full Corpus Statistics (Original vs. Selected)

The table below shows the original VoxLingua107 clip counts per duration bucket alongside the selected subset used in MLADDC. Duration buckets represent clip lengths in seconds.

| Language         | Code | Split    | 0–5s  | 5–10s  | 10–15s | 15–20s | 20s+  | Total (hrs) |
|------------------|------|----------|-------|--------|--------|--------|-------|-------------|
| Russian          | ru   | Original | 2028  | 8860   | 7044   | 5844   | 22    | 73          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.39**   |
| French           | fr   | Original | 4234  | 9465   | 6248   | 4495   | 7     | 67          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.38**   |
| Arabic           | ar   | Original | 3950  | 8422   | 5390   | 3914   | 6     | 59          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.34**   |
| Spanish          | es   | Original | 988   | 5117   | 3817   | 2941   | 3     | 39          |
|                  |      | Selected | 988   | 1004   | 1004   | 1004   | —     | **11.37**   |
| Vietnamese       | vi   | Original | 6861  | 12292  | 5169   | 3039   | 5     | 64          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.28**   |
| Mandarin Chinese | zh   | Original | 3243  | 6220   | 3861   | 3004   | 0     | 44          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.37**   |
| English          | en   | Original | 1232  | 5953   | 4824   | 3874   | 2     | 49          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.32**   |
| Hindi            | hi   | Original | 5492  | 11908  | 7382   | 5240   | 8     | 81          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.42**   |
| Portuguese       | pt   | Original | 4572  | 9725   | 5764   | 4153   | 10    | 64          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.37**   |
| Sanskrit         | sa   | Original | 2575  | 3978   | 938    | 328    | 0     | 15          |
|                  |      | Selected | 1367  | 1367   | 938    | 328    | —     | **8.92**    |
| Bahasa Indonesia | id   | Original | 3880  | 7399   | 3251   | 1980   | 0     | 40          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.35**   |
| Bengali          | bn   | Original | 4433  | 8930   | 4861   | 3195   | 0     | 55          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.36**   |
| Finnish          | fi   | Original | 913   | 4443   | 3249   | 2532   | 0     | 33          |
|                  |      | Selected | 913   | 1029   | 1029   | 1029   | —     | **11.52**   |
| Japanese         | ja   | Original | 4948  | 9262   | 4879   | 3218   | 0     | 56          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.34**   |
| Gujarati         | gu   | Original | 3766  | 7290   | 3998   | 2842   | 0     | 46          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.36**   |
| Tamil            | ta   | Original | 3743  | 7679   | 4486   | 3267   | 0     | 51          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.40**   |
| Punjabi          | pa   | Original | 4367  | 9098   | 4549   | 3078   | 0     | 54          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.36**   |
| Urdu             | ur   | Original | 1254  | 4817   | 4011   | 3571   | 0     | 42          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.35**   |
| Swedish          | sv   | Original | 2387  | 5136   | 3080   | 2174   | 0     | 34          |
|                  |      | Selected | 1000  | 1000   | 1000   | 1000   | —     | **11.39**   |
| German           | de   | Original | 885   | 4981   | 3986   | 3012   | 0     | 39          |
|                  |      | Selected | 885   | 1038   | 1038   | 1039   | —     | **11.54**   |

---

## Getting Started

### Prerequisites

Install all dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes the following packages:

| Group                    | Package       | Min. Version |
|--------------------------|---------------|--------------|
| **Deep Learning**        | torch         | ≥ 2.0.0      |
|                          | torchvision   | ≥ 0.15.0     |
|                          | torchaudio    | ≥ 2.0.0      |
| **Transformer Models**   | transformers  | ≥ 4.36.0     |
|                          | datasets      | ≥ 2.16.0     |
| **Scientific Computing** | numpy         | ≥ 1.24.0     |
|                          | scipy         | ≥ 1.11.0     |
| **Visualisation**        | matplotlib    | ≥ 3.7.0      |
|                          | seaborn       | ≥ 0.13.0     |
| **ML Utilities**         | scikit-learn  | ≥ 1.3.0      |

> **GPU Support:** For CUDA-enabled PyTorch, install it separately before running the requirements file:
> ```bash
> # Example for CUDA 11.8
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```
> Find the right command for your CUDA version at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Dataset Access

The MLADDC corpus will be made publicly available upon paper acceptance. In the meantime, please contact the corresponding author with your institutional affiliation and intended use:

**Prof. Hemant A. Patil**  
Speech Research Lab, DA-IICT, Gandhinagar, Gujarat, India  
📧 `hemant_patil@daiict.ac.in`

The original source data (VoxLingua107) is publicly available at [bark.phon.ioc.ee/voxlingua107](http://bark.phon.ioc.ee/voxlingua107/).

---

## Citation

If you use MLADDC or any findings from this work in your research, please cite *(citation will be updated upon acceptance)*:

```bibtex
@inproceedings{shah2025mladdc,
  title     = {MLADDC: Multi-Lingual Audio Deepfake Detection Corpus},
  author    = {Shah, Arth J. and Purohit, Ravindrakumar M. and
               Vaghera, Dharmendra H. and Patil, Hemant A.},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2025},
  address   = {Hyderabad, India},
  publisher = {IEEE}
}
```

---

## Related Resources

- [VoxLingua107 Dataset](http://bark.phon.ioc.ee/voxlingua107/) — Source corpus for MLADDC
- [ASVspoof Challenge](https://www.asvspoof.org/) — Leading benchmark for anti-spoofing research
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) — Audio deepfake detection dataset
- [HuBERT (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [Wav2Vec 2.0 (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)

---

## License

This repository is released for **academic and non-commercial research use only**. The MLADDC corpus is derived from VoxLingua107 and inherits its respective licensing terms. Redistribution of derived materials requires written permission from the authors.

---

*Speech Research Lab, DA-IICT, Gandhinagar, Gujarat, India*

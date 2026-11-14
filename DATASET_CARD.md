# StyleForge Dataset Card
## CelebA Authentic Photographic Face Subset ($64\times 64$)

**Dataset Name:** CelebFaces Attributes Dataset (CelebA Subset)  
**Original Source:** The Chinese University of Hong Kong (Liu et al., 2015)  
**Website:** [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
**License / Terms:** Non-commercial research and educational use only  

---

## 1. Dataset Summary

StyleForge utilizes an audited, deterministic photographic subset of the Large-scale CelebFaces Attributes (CelebA) dataset. CelebA contains over 200,000 celebrity portrait images. For the StyleForge research benchmarks, a deterministic partition of **6,140 verified authentic photographic human faces** was extracted, preprocessed, and audited for zero data leakage.

---

## 2. Dataset Partitioning & Split Manifests

All partitions are deterministically defined and cryptographically disjoint:

| Partition | Image Count | Percentage | MD5 Unique Hashes | Split File |
|---|:---:|:---:|:---:|---|
| **Training Set** | $4,912$ | $80.0\%$ | $4,912$ | `data/celeba/train_split.txt` |
| **Validation Set** | $614$ | $10.0\%$ | $614$ | `data/celeba/val_split.txt` |
| **Test Set** | $614$ | $10.0\%$ | $614$ | `data/celeba/test_split.txt` |
| **Total** | **6,140** | **100.0%** | **6,140** | Deterministic Partition |

### Leakage Audit Findings
Exhaustive cross-partition MD5 checksum comparison:
- Train / Val Overlap: **0 images (0.0%)**
- Train / Test Overlap: **0 images (0.0%)**
- Val / Test Overlap: **0 images (0.0%)**

---

## 3. Preprocessing Pipeline

1. **Source Image Resolution:** Original CelebA aligned and cropped portraits ($178\times 218$).
2. **Center Cropping:** Symmetrical crop around the facial bounding box ($178\times 178$).
3. **Resizing:** Scaled to target spatial dimensions ($64\times 64\times 3$) using bicubic anti-aliased interpolation.
4. **Color Representation:** Standard 3-channel RGB.
5. **Tensor Normalization:** Pixel values normalized to floating-point range $[-1.0, 1.0]$ for generator/critic processing:
   $$x_\text{norm} = \frac{x}{127.5} - 1.0$$

---

## 4. Considerations & Limitations

- **Commercial Restrictions:** The underlying CelebA dataset is restricted to non-commercial research and educational use. StyleForge code is licensed under MIT, but models trained on CelebA inherit the non-commercial research terms of the source data.
- **Demographic Representation:** CelebA features celebrities with heavy representation of young adults and Western celebrity aesthetics. Metrics computed on this dataset should be interpreted as fidelity to the CelebA distribution, not to the global human population.
- **Privacy Considerations:** The images represent public figures in public media contexts. No private or non-consensual biometric data is included.

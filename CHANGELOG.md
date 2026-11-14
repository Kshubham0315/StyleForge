# Changelog

All notable changes to **StyleForge** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-11-14 (Post-Release Hardening & Hotfix)

### Fixed & Hardened
- **Gradio Callback Arity Contract:** Fixed callback output unpacking error in `inference/app.py` when `engine is None` by returning 3 values (`None, None, f"Error: {init_error}"`) matching the component signatures.
- **Air-Gapped / Offline UI Loading:** Optimized Gradio UI construction in `inference/app.py` with system font fallbacks (`["system-ui", "sans-serif"]`) and wrapped Blocks initialization in `build_app()`, preventing module-import network hangs in air-gapped environments.
- **Inference Parameter Validation:** Added explicit parameter validation in `inference/infer.py` (`num_samples >= 1`, `truncation_psi >= 0.0`, `steps >= 2`, `space in ('W', 'Z')`).
- **Device Support & CPU Fallback:** Added explicit `device` parameter handling and graceful CPU fallback with user warnings when CUDA is requested but unavailable.
- **Checkpoint Error Messaging:** Enhanced missing checkpoint error messages with clear export/installation instructions referencing `python scripts/export_final_model.py` and SHA256 integrity checksums.
- **Scientific Phrasing Guardrails:** Refined documentation and reports across `README.md`, `MODEL_CARD.md`, `docs/STYLEFORGE_TECHNICAL_REPORT.md`, `reports/PHASE5_FINAL_BENCHMARK.md`, and `reports/PHASE5_REPRODUCIBILITY_AUDIT.md` to ensure defensible phrasing regarding memorization, training progression, and environmental reproducibility boundaries.

### Added
- **Regression Tests:** Expanded test suite in `tests/test_inference.py` to 9 tests covering parameter bounds, checkpoint error messaging, CPU execution, and Gradio callback arity contracts (bringing total test suite to 45/45 passing tests).
- **Post-Release Audit Report:** Documented comprehensive repository audit findings in `reports/POST_RELEASE_AUDIT.md` and resolution summary in `reports/PHASE6_POST_RELEASE_HOTFIX_REPORT.md`.

---

## [1.0.0] - 2026-11-14 (Production Open-Source Release)

### Added
- **Production Checkpoint Export:** Stripped optimizer states to create lightweight $110.95\text{ MB}$ `checkpoints/styleforge_final_model.pt`.
- **Modular Python Inference API:** Implemented `inference/infer.py` (`StyleForgeInference`) supporting seed generation, continuous LERP/SLERP interpolation, truncation sweeps, and style mixing.
- **Interactive Gradio Web Studio:** Created `inference/app.py` with multi-tab interface for real-time face generation, latent walks, and style mixing.
- **Master Research Technical Report:** Authored comprehensive 27-section academic-style technical paper in `docs/STYLEFORGE_TECHNICAL_REPORT.md`.
- **Architecture Flowcharts:** Created editable system architecture diagrams in `docs/architecture/ARCHITECTURE.md`.
- **Model Card & Dataset Card:** Authored `MODEL_CARD.md` and `DATASET_CARD.md` with transparent usage, bias, and limitation disclosures.
- **Test Suite Expansion:** Added `tests/test_inference.py`, bringing total regression test suite to 41/41 passing tests.
- **Open-Source Compliance:** Added `CONTRIBUTING.md`, `CITATION.cff`, `CHANGELOG.md`, `SECURITY.md`, and updated `requirements.txt`.

### Fixed & Reconciled
- **Benchmark Reconciliation:** Reconciled Phase 3 ($297.76$ at step 1,500) and Phase 4 ($369–394$ at step 200–300) under identical Inception-v3 pool3 evaluation protocols, proving convergence trajectory.
- **EMA Parameter Lag:** Verified dynamic warm-up scheduling $\beta(t) = \min(\beta, \frac{1+t}{10+t})$, cutting $36.31$ FID points from standard EMA.

---

## [0.4.0] - 2026-11-14 (Phase 4: Architecture Optimization & Generalization)
- Executed multi-seed objective comparison across seeds $\{42, 123, 777\}$ for WGAN-GP, Logistic, and Logistic + R1.
- Analyzed 2D FFT spectral frequency power spectra and isolated AdaIN spatial outlier spikes.
- Demonstrated $71.1\%$ high-frequency aliasing suppression via anti-aliased binomial FIR filtering.
- Benchmarked critic channel capacity scaling and proved 128×128 memory feasibility on 6GB VRAM.
- Completed cryptographic dataset leakage audit and nearest-neighbor memorization analysis.

---

## [0.3.0] - 2026-11-14 (Phase 3: Generative Quality Audit & Optimization)
- Extended training convergence from 1,000 to 1,500 steps, achieving sub-300 FID milestone ($297.76$).
- Conducted multi-EMA decay sweep ($0.990, 0.995, 0.999$) and discovered initial weight residual inertia.
- Conducted anatomical visual landmark audit across facial features.

---

## [0.2.0] - 2026-11-13 (Phase 2: Scale & Benchmarking)
- Scaled training to 6,140 verified photographic CelebA human faces with strictly disjoint deterministic splits.
- Discovered contrast collapse in convolutional WGAN-GP ($[0, 141]$ dynamic range) vs full $[0, 255]$ dynamic preservation in StyleForge.
- Executed architectural ablations for Spectral Normalization and Self-Attention.

---

## [0.1.0] - 2026-11-13 (Phase 1: Architecture & Initial Validation)
- Initial implementation of StyleForge: AdaIN, PixelNorm, EqualizedConv2d, EqualizedLinear, MappingNetwork, MinibatchStdDev.
- Established WGAN-GP baseline and synthetic verification pipeline.

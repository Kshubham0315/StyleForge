# Contributing to StyleForge

Thank you for your interest in contributing to **StyleForge**! We welcome contributions that improve generative quality, mathematical rigor, training efficiency, documentation clarity, and test coverage.

---

## Code of Conduct

Please maintain a collaborative, respectful, and scientifically honest environment. Never submit fabricated benchmark metrics or synthetic visual samples that do not match committed checkpoints.

---

## Development Workflow

1. **Fork and Clone:**
   ```bash
   git clone https://github.com/Kshubham0315/StyleForge.git
   cd StyleForge
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Branching Strategy:**
   Create a descriptive feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   # or fix/your-bugfix-name
   ```

4. **Code Quality & Integrity:**
   - Adhere to PEP 8 styling.
   - Maintain pure FP32 calibration for sensitive mapping and synthesis layers.
   - All contributions modifying models, losses, or data loaders **must include corresponding unit tests** in `tests/`.

5. **Run Test Suite:**
   Before submitting a Pull Request, confirm that all regression tests pass:
   ```bash
   pytest tests/ -v
   ```

6. **Submit a Pull Request:**
   Provide a clear summary of your changes, reference any relevant issues or reports, and verify that all automated tests pass.

---

## Reporting Issues

If you discover a bug, numerical instability, or metric calculation discrepancy, please open an Issue with:
- Exact environment details (OS, PyTorch version, CUDA version, GPU model).
- Minimal reproducible code snippet.
- Full error traceback.

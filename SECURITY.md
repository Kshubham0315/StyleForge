# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

StyleForge takes model safety, dependency security, and code integrity seriously.

If you discover a security vulnerability (such as arbitrary code execution during checkpoint loading, deserialization vulnerabilities, or credential exposures):

1. **Do not disclose the vulnerability publicly** in issues, pull requests, or discussions.
2. Email the maintainer directly at **kshubham0315@gmail.com** with:
   - Type of vulnerability.
   - Proof-of-concept script or steps to reproduce.
   - Potential impact.
3. You will receive an acknowledgment within 48 hours. Valid vulnerabilities will be patched promptly with an advisory credit.

---

## Safe Model Loading Notice

PyTorch checkpoints utilize Python's pickle module during deserialization. Always ensure you load weights exclusively from verified, official StyleForge releases or trusted sources. In StyleForge v1.0, the production checkpoint `checkpoints/styleforge_final_model.pt` contains strictly parameter dictionaries and numerical metadata.

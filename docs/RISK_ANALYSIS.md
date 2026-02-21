# Risk Analysis (ISO 14971–oriented)

This is a simplified risk analysis for portfolio/educational context. The system is not a certified medical device.

## Intended Use

Research/demonstration: brain MRI denoising using a U-Net trained on synthetic noise. Not for direct clinical diagnosis or treatment decisions.

## Hazard Identification and Risk Control

| Hazard | Cause | Severity | Likelihood | Mitigation |
|--------|--------|----------|------------|------------|
| Incorrect or degraded output | Model error, wrong checkpoint, bad input | Medium | Possible | Unit and smoke tests; validation metrics (PSNR/SSIM); human review of sample outputs. |
| Misuse as sole diagnostic tool | User assumes output is clinically validated | High | Low | Clear disclaimer: research/demo only; not for clinical use; no FDA clearance claimed. |
| Data leakage / PHI | Real patient data in repo or logs | High | Low | No PHI in repo; synthetic noise only; data handling note in README. |
| Service unavailability | API down, container failure | Low | Possible | Logs; restart policy; deployment docs. |
| Dependency vulnerability | Outdated or vulnerable libraries | Medium | Possible | Pinned requirements; periodic review; CI on dependencies. |

## Residual Risk

Accepted for a non–clinical, demonstrator system with disclaimers and no claim of regulatory compliance. For a real SaMD, a full QMS (e.g. ISO 13485, IEC 62304) and documented risk management would be required.

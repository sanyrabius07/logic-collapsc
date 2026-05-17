# Threat Model — Logic Collapse Horizon

## Overview

This document formally defines the threat model for Logic Collapse,
as required by the ACM CCS 2026 Open Science appendix.

---

## Four-Component CCS Threat Model

### (i) Deployer / Threat Actor
A network security engineer or DevOps practitioner who applies
standard model compression (PTQ, pruning, Knowledge Distillation)
to a trained IDS neural network for edge deployment.

- No malicious intent is required
- Uses default framework APIs (PyTorch, TensorFlow)
- Has full access to model weights and training data
- Monitors model quality using accuracy and F1 score only

### (ii) Threat Surface
The **explanation layer** of the compressed model.

Post-compression, SHAP attribution vectors diverge from the teacher's
ground truth reasoning while ACC/F1 remain within 0.5% of the original —
below any operational monitoring threshold.

The surface is **invisible to accuracy-based auditing**.

### (iii) Generality
Logic Collapse is demonstrated across:
- 4 neural architectures: MicroMLP, TinyCNN, ResidualMLP, MiniTabTransformer
- 6 compression families: PTQ-4bit, PTQ-8bit, Pruning-30%, Pruning-70%, KD, VanillaLoRA
- 4 IDS datasets: Phishing (N=11K), NSL-KDD (N=151K), RT-IoT2022 (N=123K), UNSW-NB15 (N=258K)
- Total: 542,010 network traffic samples

The phenomenon is not model-specific or dataset-specific.

### (iv) Practicality
- All compression methods use production-standard defaults
- No specialised capability required beyond PyTorch
- KD — the most widely deployed compression method in edge IDS —
  induces the **worst** Logic Collapse (LCI = 0.585, p < 0.001)
- PTQ-4bit — widely used for MCU deployment — displaces the critical
  security feature from rank 1 to rank 4 (MPRF = 4)

---

## Security Impact

| Impact | Description |
|---|---|
| False triage priority | Wrong features flagged as critical, delaying SOC response |
| Missed forensic attribution | Attack signatures masked in post-incident analysis |
| Regulatory violation | EU AI Act Art. 13 compliance claims based on corrupted explanations |
| Silent degradation | No accuracy alarm fires — undetectable without LCI monitoring |

---

## Out of Scope

- Adversarial attacks by external actors on the network
- Packet-level (image-like) IDS representations
- Multi-class attack classification (binary only in this paper)
- Graph-based network intrusion detection

# Detector: data poisoning

**Emits:** `data.poisoning.*`. **Threat model v1:** non-adaptive attackers; state this limit in reports.

## Methods
| Threat | Method |
|---|---|
| Label flipping / noise | cleanlab confident learning with cross-validated probabilities |
| Backdoor / trigger (image, embedding) | Fit a small proxy model; run Spectral Signatures and Activation Clustering on penultimate features |
| Clean-label | Feature-space outlier + activation clustering; flag clusters with label/feature mismatch |
| Injected/OOD rows (tabular) | Isolation Forest / ECOD fit on known-good batches |
| Source-level | Per-source batch profiling; two-sample tests vs. history; trust score per source |
| Source flooding | Near-duplicate rate and per-source volume anomaly |
| Text triggers | Rare n-gram/phrase frequency anomaly correlated with label |
| Attribution (v2) | TracIn / gradient similarity from bad prediction to training rows |

## Design
- Output a ranked suspect list with scores, plus a recommended quarantine set. Never delete data automatically (always Tier 2).
- Proxy model must be trained on a split disjoint from the one being audited when feasible.
- Report calibration: FPR at the chosen threshold on clean data.

## Targets
≥85% recall at ≤5% FPR on 5% label-flip and BadNets-style backdoors. Report degradation for stealthier attacks. Datasets: MNIST/CIFAR-10 with ART/BackdoorBench attacks, UCI Adult/Credit Fraud with injected rows, DVP PO-*.

# Research history: Gates 1-33

This file records why the v1.0 release is smaller than the research tree.

| Gate | Main result |
|---:|---|
| 1 | Exact oracle gradient antidote validated; exact trajectory recovery. |
| 2A | Learned trusted-only reconstruction worked on Digits. |
| 2B | Fashion-MNIST transfer improved poisoned gradients and accuracy. |
| 3 | Extreme poisoning exposed need for abstention/risk control. |
| 4 | Identifiability boundary: stealth corruption can be indistinguishable from legitimate ambiguity. |
| 5 | Multiview/temporal evidence shrank stealth region. |
| 6 | Empirical resistance-radius/stealth-capacity analysis. |
| 7 | Finite-set last-layer certificate. |
| 8 | Full-network fixed-state gradient certificate. |
| 9 | Frozen-representation trajectory certificate. |
| 10 | Nonlinear trainable trajectory certificate. |
| 11 | Capacity/certifiability scaling. |
| 12 | Accuracy/certification frontier. |
| 13 | Certificate-aware co-design and fail-closed selection. |
| 14 | Differentiable certificate-constrained training. |
| 15 | Blockwise certificate-budget allocation. |
| 16 | Sample-conditional routing improved risk suppression but not reliable utility. |
| 17 | Trusted-evidence routing improved over task-coordinate routing. |
| 18 | Temporal evidence reduced ambiguity but candidate sets were not label oracles. |
| 19 | Simultaneous conformal candidate sequence (SCCS). |
| 20 | Antidote controller removed the raw untrusted label causal path exactly. |
| 21 | Trusted graph teacher + SCCS+ recovered utility while preserving invariance. |
| 22 | Fashion transfer passed; CIFAR exposed representation insufficiency. |
| 23 | Pretrained MobileNet representation diagnostic improved separability but was not kept for open core. |
| 24 | Target-domain linear SSL failed to rescue teacher geometry. |
| 24B | Open CPU patch encoder rescued CIFAR partially; corrected earlier preprocessing mismatch. |
| 25 | Graph/LDA/proprietary-geometry branch rejected; open core simplified. |
| 26 | Fixed HOG + spatial-color trusted teacher produced a large open representation gain. |
| 27 | Confidence routing/weighting failed out of sample; direct soft targets retained. |
| 28 | Probability calibration improved NLL/ECE but not student utility; rejected from target path. |
| 29 | 256 random RBF landmarks recovered about one third of teacher/student gap. |
| 30 | Smarter geometric landmark selection failed to reliably beat random; random retained. |
| 31 | Full 50k/10k centralized qualification passed; high-trust utility crossover documented. |
| 32 | Full 50k/10k federated/non-IID qualification passed; final-student mutation invariance measured directly. |
| 33 | External FairMean/FedAvg/q-FFL checkpoint confirmed robustness/utility tradeoff; full GPU matrix left optional. |

## Removed from the v1.0 core

- VIVERE / MACSL / Ghost Manifold components;
- geometric poison classification;
- graph diffusion;
- trusted LDA metric layer;
- mandatory SCCS projection;
- confidence filtering and sample weighting;
- pseudo-target probability recalibration;
- non-random landmark selection.

These exclusions are deliberate research conclusions, not missing implementation
work.

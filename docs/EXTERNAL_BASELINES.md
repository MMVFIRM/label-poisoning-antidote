# External baseline qualification

Gate 33 intentionally avoids fake reimplementations of methods whose defining
behavior depends on deep neural-network training dynamics.

## FairMean

Reference paper:

> H. Zheng, J. Zhang, Y. Liu, "FairMean: Promoting Fairness in Distributed
> Learning under Label Poisoning Attacks," arXiv:2609.26377v1, 2026.

Public repository used by the optional reproduction scripts:

`https://github.com/Zhg9300/FairnessUnderLP`

The matched Gate-33 checkpoint used the FairMean bounded marginal weight

```text
a(z) = 1 + kappa*z/(z+tau),  kappa=tau=1
```

in a shared fixed-feature model class. It showed FairMean preserving more
accuracy than LPA when many raw labels remain clean, while remaining sensitive
to increasing corruption because the poisoned labels still enter its objective.

## Co-teaching and DivideMix

They were not replaced by ridge heuristics. Co-teaching relies on two networks
exchanging small-loss examples; DivideMix uses two networks, mixture modeling,
and semi-supervised training. A publication claiming direct comparison should
run their original implementations or another faithful implementation.

## Confident Learning / cleanlab

A faithful comparison should use out-of-sample predicted probabilities and the
published confident-learning procedure. A hand-written disagreement filter is
not labeled as cleanlab in this repository.

## Gate-33 status

External qualification is **partial**. The scientific conclusion currently
supported is a robustness/utility tradeoff, not overall superiority to the
external methods.

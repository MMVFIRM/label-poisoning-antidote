# Threat model

## Supported threat

The validated claim applies when:

1. A subset of labels is explicitly designated **trusted**.
2. Every other supplied label is designated **untrusted**.
3. An attacker may replace the untrusted label values arbitrarily.
4. Input features/examples themselves are unchanged.
5. Trusted set membership and trusted labels are not controlled by the attacker.
6. The supplied LPA training path is used without adding a side channel from
   untrusted labels into feature extraction, target construction, landmark
   selection, hyperparameter selection, stopping criteria, or model selection.

Under those conditions the untrusted label field is absent from the objective,
so arbitrary changes to that field produce no change in the LPA targets or
student solution except possible numerical nondeterminism introduced outside
the validated implementation.

## Explicitly out of scope

LPA v1.0 does not claim protection from:

- corruption of trusted labels;
- manipulation of trusted-set membership;
- feature/input poisoning;
- sample insertion, deletion, duplication, or reordering attacks that alter the
  feature dataset itself;
- arbitrary Byzantine model updates or executable client code;
- availability attacks or malicious client non-participation;
- privacy inference from client sufficient statistics;
- compromised random-number generation;
- poisoned external/pretrained feature extractors;
- a malicious teacher implementation;
- attacks that alter hyperparameters or deployment policy.

## Trusted-label contamination

Gate 31 deliberately corrupted trusted labels at one checkpoint. Small trusted
contamination produced modest degradation; 10-20% trusted corruption caused
substantial accuracy loss. That experiment is a stress diagnostic, not a new
robustness guarantee.

## Federated assumption

The qualified federated architecture uses a **global trusted root** and shared
landmark basis. Clients do not independently construct semantic teachers.

A fully decentralized system in which each client must build its own teacher is
a different architecture and is not covered by the Gate-32 evidence.

## Identifiability boundary

Invariance is not the same as correctness. If the true class cannot be inferred
from `x` and trusted evidence, LPA cannot manufacture the missing information.
The method can make poisoned labels irrelevant while still having low predictive
accuracy when the trusted semantic root is too small or unrepresentative.

#!/usr/bin/env python3
"""Small synthetic demo of the LPA untrusted-label invariant."""

import json

import numpy as np

from lpa import BlendedKernelTeacher, KMeansTeacherConfig, TeacherConfig, mutation_invariance_audit

rng=np.random.default_rng(1)
n=240
classes=4
y=np.repeat(np.arange(classes),n//classes)
rng.shuffle(y)
centers_a=rng.normal(size=(classes,10))*1.5
centers_b=rng.normal(size=(classes,6))*1.2
a=centers_a[y]+.65*rng.normal(size=(n,10))
b=centers_b[y]+.65*rng.normal(size=(n,6))
a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
z=np.concatenate([a,b],axis=1)/np.sqrt(2.)
trusted=np.concatenate([np.flatnonzero(y==c)[:10] for c in range(classes)])

# LPA 2.0 default teacher: joint-feature kernel blended with the two-view teacher.
teacher=BlendedKernelTeacher(classes,KMeansTeacherConfig(gamma=1.),TeacherConfig(gamma_view_a=1.,gamma_view_b=1.))
teacher.fit(a[trusted],b[trusted],z[trusted],y[trusted])
p=teacher.predict_proba(a,b,z)

observed_a=y.astype(object)
observed_b=y.astype(object)
untrusted=np.ones(n,dtype=bool);untrusted[trusted]=False
observed_b[untrusted]="POISONED_ARBITRARY_VALUE"

# Audits the default LPA 2.0 linear student; pass a StudentConfig for v1.0.
report=mutation_invariance_audit(p,z,trusted,observed_a,observed_b,classes)
print(json.dumps(report.to_dict(),indent=2))

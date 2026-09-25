import numpy as np

from lpa.audit import mutation_invariance_audit
from lpa.config import StudentConfig, TeacherConfig
from lpa.teacher import TrustedKernelTeacher


def make_problem(seed=7):
    rng=np.random.default_rng(seed)
    n=160
    classes=4
    y=np.repeat(np.arange(classes), n//classes)
    rng.shuffle(y)
    ca=rng.normal(size=(classes,8))*1.5
    cb=rng.normal(size=(classes,5))*1.3
    a=ca[y]+.6*rng.normal(size=(n,8))
    b=cb[y]+.6*rng.normal(size=(n,5))
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    z=np.concatenate([a,b],axis=1)/np.sqrt(2.)
    trusted=[]
    for c in range(classes):
        trusted.extend(np.flatnonzero(y==c)[:8])
    return a,b,z,y,np.asarray(trusted,dtype=np.int64),classes


def test_untrusted_label_mutation_changes_neither_targets_nor_student():
    a,b,z,y,trusted,classes=make_problem()
    teacher=TrustedKernelTeacher(classes,TeacherConfig(gamma_view_a=1.,gamma_view_b=1.))
    teacher.fit(a[trusted],b[trusted],y[trusted])
    p=teacher.predict_proba(a,b)
    labels_a=y.astype(object)
    labels_b=y.astype(object)
    mask=np.ones(len(y),dtype=bool);mask[trusted]=False
    labels_b[mask]="UNTRUSTED_DO_NOT_READ"
    report=mutation_invariance_audit(
        p,z,trusted,labels_a,labels_b,classes,
        StudentConfig(landmark_count=24,landmark_seed=11,landmark_gamma=.5,ridge=1.),
    )
    assert report.passed
    assert report.target_hash_a==report.target_hash_b
    assert report.weight_hash_a==report.weight_hash_b

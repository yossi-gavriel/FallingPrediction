"""falls_ml — clinical fall-risk prediction pipeline.

Scientific baseline: reproduction of the published eFalls model
(Archer et al., Age Ageing 2024;53(3):afae057). See docs/EFALLS_REPRODUCTION_SPEC.md.

Public entry points are imported lazily to keep ``import falls_ml`` cheap:

    from falls_ml.experiment import run_experiment
    from falls_ml.inference import predict_risk
"""

__version__ = "0.12.2"

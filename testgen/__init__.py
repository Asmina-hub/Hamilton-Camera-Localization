"""Synthetic "hidden test run" generator: crop a lap, strip the ground truth, apply controlled perturbations.

    python3 scripts/make_testset.py BAG gt.csv --lap 6 --spec "speedup:k=1.25+lamps_off" --out out/testsets/x

Every perturbation is a separate transform with its own parameters (testgen/transforms.py); a spec combines
any of them with '+'. Named presets live in testgen/presets.py and can be used inside a spec as '@name'.
"""

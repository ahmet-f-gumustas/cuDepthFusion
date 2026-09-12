"""Dataset download, manifests and adapters.

Filter inputs (noisy depth, intrinsics, poses) and evaluator ground truth (clean depth)
are opened through separate entry points so the filter can never see clean depth.
"""

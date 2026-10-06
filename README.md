CentriQ is a small JAX implementation of the calibration-free quantizer from arXiv:2610.06260.
pip install -e .
python3 -m pytest
This is the centering, Hadamard rotation, rank-1 mean restore, and grouped ell_p weight fit. It is not a diffusion transformer, not a fused kernel, and not the reported image metrics.

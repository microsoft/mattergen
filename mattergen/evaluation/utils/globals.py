# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Default threshold (eV/atom above hull) we use to consider a structure stable
DEFAULT_STABILITY_THRESHOLD = 0.1
# Increased RMSD threshold used for structure matching in the context of RMSD metrics
# This increased cutoff is needed to get an atom alignment even in case structures don't match
MAX_RMSD = 0.5

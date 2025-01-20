# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class IncompatibleSampler(ValueError):
    # Raised when sampler type and SDE are incompatible.
    pass


class AmbiguousConfig(ValueError):
    # Raised when the config is ambiguous
    pass

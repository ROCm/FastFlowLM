# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""SmolVLA on the AMD NPU -- python distribution of the prebuilt extension.

The compiled extension is shipped inside this package directory rather than at
the site-packages root, so that the wheel owns every file it installs and
uninstalls cleanly. The top-level ``smolvla_npu`` shim module loads it from
here; see ``_extension.py``.

Import the NPU model as::

    import smolvla_npu
    model = smolvla_npu.SmolVLA("/path/to/SmolVLA")
"""

__version__ = "0.1.0"

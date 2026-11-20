#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.
# Portions of this file consist of AI-generated content.

"""Route an IRON layer.mlir the way aiecc does, in seconds, and run check_routes.py on it.

aiecc routes after its address pipeline (which also adds its own control-packet
flows), so routing the plain IRON MLIR gives a different answer. This runs the
same pipeline (INPUT_WITH_ADDRESSES_PIPELINE) and then aie-opt's pathfinder.

The router's result changes from run to run (its choices depend on memory
addresses, randomized by Linux), so one run says only what one routing looks
like; run it several times to see how often a design passes. It does not predict
a given aiecc build, which the build checks itself (makefiles/bitstream.mk).

Usage: quick_route.py <iron layer.mlir>     (exit code = check_routes.py's)
"""
import os
import subprocess
import sys
import tempfile

from aie.compiler.aiecc.main import INPUT_WITH_ADDRESSES_PIPELINE, run_passes

layer_mlir = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
with tempfile.TemporaryDirectory() as tmp:
    with_addresses = os.path.join(tmp, "input_with_addresses.mlir")
    physical = os.path.join(tmp, "input_physical.mlir")
    # same arguments aiecc uses by default for this design (see makefiles/bitstream.mk)
    pipeline = INPUT_WITH_ADDRESSES_PIPELINE("", False, False, False, "AIE2P").materialize(module=True)
    run_passes(pipeline, open(layer_mlir).read(), with_addresses, False)
    subprocess.run(["aie-opt", "--aie-create-pathfinder-flows", with_addresses, "-o", physical], check=True)
    sys.exit(subprocess.run([sys.executable, os.path.join(here, "check_routes.py"), physical, layer_mlir]).returncode)

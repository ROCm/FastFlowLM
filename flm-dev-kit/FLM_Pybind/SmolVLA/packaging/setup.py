# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

"""Force a platform-specific wheel tag.

Everything else about this distribution is declared in pyproject.toml. This
file exists only to correct the wheel tag, which setuptools gets wrong here:
the package has no ext_modules (the extension is prebuilt and shipped as
package data), so setuptools concludes the wheel is pure python and tags it
``py3-none-any``.

That tag is a lie with real consequences -- pip would install this wheel on
aarch64, or on macOS, and only fail later at import with an unhelpful loader
error. Tagging it ``cp312-cp312-linux_x86_64`` makes pip refuse up front with a
message that names the incompatibility.

``requires-python = "==3.12.*"`` in pyproject.toml already blocks other python
versions, but says nothing about platform or ABI; this covers the rest.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that reports itself as platform-specific.

    ``has_ext_modules`` is what bdist_wheel consults to decide between a pure
    and a platform wheel. Ours is True despite there being no ext_modules to
    build, because the built artifact is already in the tree.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(distclass=BinaryDistribution)

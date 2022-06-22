################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import site
import sys
import warnings

import setuptools

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    warnings.warn("Unable to import extras")
    extras = {}

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qe-cirq",
    use_scm_version=True,
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Cirq package for Orquestra.",
    url="https://github.com/zapatacomputing/qe-cirq ",
    packages=setuptools.find_packages(where="src/python"),
    package_dir={"": "src/python"},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "z-quantum-core",
        # There's an upper bound on cirq-* libraries to guard against possible backward
        # incompatibilities in future 0.* versions.
        "cirq-core<=0.13",
        "cirq-google<=0.13",
    ],
    extras_require=extras,
)

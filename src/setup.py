import setuptools
import os

setuptools.setup(
    name="qe-cirq",
    version="0.1.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Cirq package for Orquestra.",
    url="https://github.com/zapatacomputing/qe-cirq ",
    packages=setuptools.find_namespace_packages(include=['zquantum.*']),
    package_dir={'' : 'python'},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'z-quantum-core', 'qe-openfermion',
    ]
)

import setuptools

setuptools.setup(
    name="qe-cirq",
    use_scm_version=True,
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Cirq package for Orquestra.",
    url="https://github.com/zapatacomputing/qe-cirq ",
    packages=setuptools.find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    setup_requires=["setuptools_scm~=6.0"],
    install_requires=[
        "z-quantum-core",
        "cirq-core<=0.13",
        "cirq-google<=0.13",
    ],
)

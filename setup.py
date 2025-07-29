from setuptools import setup, find_packages

setup(
    name="BesselML",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "sympy>=1.8",
        "pyoperon>=0.3.0",
        "scikit-learn>=0.24.0",
        "optuna>=2.10.0",
        "ipython>=7.0.0",
    ],
    author="Daniel C.",
    author_email="dan.ctvrta@email.cz",
    description="A symbolic regression framework for mathematical functions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Danyanne/BesselML",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)

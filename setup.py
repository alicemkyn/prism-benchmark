from setuptools import setup, find_packages

setup(
    name="prism-bench",
    version="0.1.0",
    description="PRISM: Pathology Reliability In Scarce-label Medicine benchmark",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "scipy>=1.7",
    ],
    entry_points={
        "console_scripts": [
            "prism=prism_bench.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)

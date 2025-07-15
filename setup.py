from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SoftPQ",
    version="0.1.0",
    author="Ranit Karmakar",
    description="SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rkarmaka/SoftPQ",
    license="MIT",
    packages=find_packages(include=["metrics*", "data*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "scikit-image",
        "pandas"
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "softpq-eval=scripts.run_eval:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    extras_require={
        "dev": ["pytest"]
    },
)

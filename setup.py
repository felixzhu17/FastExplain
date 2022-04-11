from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="modelflow",
    version="0.0.14",
    license="MIT",
    author="Felix Zhu",
    author_email="zhu.felix@outlook.com",
    description="ML flows from start to finish",
    long_description=long_description,
    packages=find_packages(),
    url="https://github.com/felixzhu17/ModelFlow",
    install_requires=[
        "interpret",
        "pandas",
        "plotly",
        "scikit_learn",
        "scipy",
        "plotly_express",
        "xgboost",
        "statsmodels",
        "shap",
        "numpy==1.21",
        "pytest",
    ],
)

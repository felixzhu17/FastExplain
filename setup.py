from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="fast-explain",
    version="0.0.1",
    license="MIT",
    author="Felix Zhu",
    author_email="zhu.felix@outlook.com",
    description="Fast Fit, Fast Explain",
    long_description=long_description,
    packages=find_packages(),
    url="https://github.com/felixzhu17/FastExplain",
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

from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as f:
    required_packages = [ln.strip() for ln in f.readlines()]

setup(
    name="Predictive Maintenance",
    version="0.0.1",
    description="Predictive Maintenance Multi-Model Classification Project",
    author="Vishwas Gowda",
    url="https://github.com/vishwasg217/Predictive-Maintenance",
    packages=find_namespace_packages(),
    install_requires=required_packages,
)

# setup.py
docs_packages = ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]


# setup.py
style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]

# Define our package
setup(
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages,
        "docs": docs_packages,
    },
)


setup(
    extras_require={
        "dev": docs_packages + style_packages,
        "docs": docs_packages,
    },
)

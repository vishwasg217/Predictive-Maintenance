from pathlib import Path
from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, 'requirements.txt'), 'r') as f:
    required_packages = [ln.strip() for ln in f.readlines()]

setup(
    name='Predictive Maintenance',
    version='0.0.1',
    description='Predictive Maintenance Multi-Model Classification Project',
    author='Vishwas Gowda',
    url='https://github.com/vishwasg217/Predictive-Maintenance',
    packages=find_namespace_packages(),
    install_requires=required_packages,
)
"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from pathlib import Path
from setuptools import setup

ROOT_DIR = Path.cwd().parent


with open(ROOT_DIR / "README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="src",
    version="0.0.1",
    package_data={},
    packages=["trams"],
    description="NeuronSW trams classification project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "train_model=trams.train_model:main",
        ]
    },
)

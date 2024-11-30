from setuptools import find_packages, setup

setup(
    name="darts-toolkit",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="0.1.0",
)

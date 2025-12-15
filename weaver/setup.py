from setuptools import setup, find_packages

setup(
    name="weaver-local",
    version="0.0.0",
    description="Local editable install for weaver during development",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
)

from setuptools import setup, find_packages

setup(
    name="petdal_network",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",  # List all dependencies required for your module to work
    ],
    author="Petter Dalhaug",
    author_email="petter.dalhaug00@outlook.com",
    description="A simple neural network package for educational purposes.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/petdalh/IT3030-DL-Assigment",
    license="MIT",
)

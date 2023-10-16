from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name="gpro",
    version="0.1.0",
    author="Qixiu Du, Haochen Wang",
    author_email="",
    description="Gpro package in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
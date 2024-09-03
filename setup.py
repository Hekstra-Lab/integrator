from setuptools import setup, find_packages

setup(
    name="integrator",
    version="0.1.0",
    description="A package for integrating X-ray diffraction data with variational inference.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Luis Aldama, Kevin Dalton",
    author_email="luis_aldama@g.harvard.edu, kmdalton@slac.stanford.edu",
    url="https://github.com/Hekstra-Lab/integrator/tree/integrator_mvn",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
  #      "dials==3.17.dev0",
        "matplotlib==3.8.2",
        "numpy>=1.21,<2.0",  # Use a version of numpy compatible with matplotlib
        "polars==1.6.0",
        "pytorch_lightning==2.2.1",
        "rs_distributions==0.0.2",
        "scipy==1.14.1",
        "setuptools==69.0.3",
        "torch==2.1.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

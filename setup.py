from setuptools import setup, find_packages


# Get version number
def getVersionNumber():
    with open("integrator/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

PROJECT_URLS = {}


LONG_DESCRIPTION = """
"""

setup(
    name="integrator",
    version=__version__,
    author="Kevin M. Dalton",
    author_email="kmdalton@fas.harvard.edu",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description="Merging crystallography data without much physics.",
    project_urls=PROJECT_URLS,
    python_requires=">=3.8,<3.12",
    url="https://github.com/kmdalton/integrator",
    install_requires=[
        "reciprocalspaceship>=0.9.16",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
    scripts=[],
    entry_points={"console_scripts": []},
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "pytest-xdist>=3"],
)

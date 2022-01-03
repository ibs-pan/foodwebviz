import sys
from setuptools import setup

if sys.version_info[:2] < (3, 7):
    sys.stderr.write(f'foodwebviz requires Python 3.7 or later ({sys.version_info[:2]} detected).\n')
    sys.exit(1)


name = "foodwebviz"
description = "Python package for creating and visualizing foodwebs"
authors = {
    "Pawluczuk": ("Łukasz Pawluczuk", ""),
    "Iskrzyński": ("Mateusz Ikrzyński", ""),
}
maintainer = ""
maintainer_email = ""
url = ""
project_urls = {
    "Bug Tracker": "https://github.com/lpawluczuk/foodwebviz/issues",
    "Documentation": "https://github.com/lpawluczuk/foodwebviz",
    "Source Code": "https://github.com/lpawluczuk/foodwebviz",
}
platforms = ["Linux", "Mac OSX", "Windows", "Unix"]
keywords = [
    "foodwebs",
]
classifiers = [  # TODO
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]


with open("foodwebviz/__init__.py") as fid:
    for line in fid:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break

packages = [
    "foodwebviz",
    "foodwebviz.animation"
]


def parse_requirements_file(filename):
    with open(filename) as f:
        requires = [x.strip() for x in f.readlines() if not x.startswith("#")]
    return requires


install_requires = parse_requirements_file("requirements.txt")

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(
        name=name,
        version=version,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        author=authors["Pawluczuk"][0],
        author_email=authors["Pawluczuk"][1],
        description=description,
        keywords=keywords,
        long_description=long_description,
        platforms=platforms,
        packages=packages,
        url=url,
        project_urls=project_urls,
        classifiers=classifiers,
        install_requires=install_requires,
        python_requires=">=3.7",
        zip_safe=False,
    )

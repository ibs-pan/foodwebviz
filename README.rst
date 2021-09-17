foodwebviz
==========

foodwebviz is a Python package for the visualization of food webs (trophic networks).

- **Source:** https://github.com/ibs-pan/foodwebviz
- **Bug reports:** https://github.com/ibs-pan/foodwebviz/issues


Installation
------------
Make sure you have Python installed (we recommend Anaconda which comes with a wide range of handy default packages, along with Jupyter Notebooks and convenient Spyder IDE: https://www.anaconda.com/).

1. Install orca first: `npm install -g electron@6.1.4 orca`
2. To create animations, install ImageMagick: https://docs.wand-py.org/en/0.3.5/guide/install.html (on Linux: 'sudo apt-get install libmagickwand-dev')
3. Manually download ``foodwebviz`` package from `GitHub <https://github.com/lpawluczuk/foodwebviz>`_ and run the following terminal command from the
top-level source directory (on Windows use e.g. Anaconda Prompt):

    $ pip install .


Tutorial
--------
``examples/visualization.ipynb`` is an interactive Jupyter Notebook with code examples and functionality overview
``examples/sample_output`` contains examples of visualisations (screenshots of interactive heatmap and graph visualisations)
To get information on a specific function/method "function_name" please execute "help(function_name)".

You can play with the tutorial Jupyter Notebook without installing the package locally: https://mybinder.org/v2/gh/ibs-pan/foodwebviz/master?filepath=examples%2Fvisualization.ipynb

Testing
-------
To execute the tests run:
$ pytest 

foodwebviz uses the Python ``pytest`` testing package.  You can learn more
about pytest on their `homepage <https://pytest.org>`_.

Bugs \ Questions
-------

In case of questions / problems please report them [here](https://github.com/ibs-pan/foodwebviz/issues)

Please report any bugs that you find `here <https://github.com/lpawluczuk/foodwebviz/issues>`_.


Citation
-------

When using foodwebviz package please cite:

# article in submission

A BibTeX entry for LaTeX users is

# article in submission

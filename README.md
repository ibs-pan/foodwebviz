foodwebviz
==========

foodwebviz is a Python package for the visualization of food webs (trophic networks).

- **Source:** https://github.com/ibs-pan/foodwebviz
- **Bug reports:** https://github.com/ibs-pan/foodwebviz/issues

Gallery
-------
![Heatmap of carbon flows over the continental shelf north of Richards Bay, South Africa](https://github.com/ibs-pan/foodwebviz/blob/master/examples/sample_output/Heatmap_Richards_Bay.png)

![Screenshot of an interactive graph visualisation of the Prince William Sound food web, Alaska](https://github.com/ibs-pan/foodwebviz/blob/master/examples/sample_output/Graph_Prince_William_Sound_Alaska.png)

![Animated flow network of Richards Bay food web](https://github.com/ibs-pan/foodwebviz/blob/master/examples/sample_output/Animation_Richards_Bay_South_Africa.gif)



Installation
------------
Make sure you have Python installed (we recommend Anaconda which comes with a wide range of handy default packages, along with Jupyter Notebooks and convenient Spyder IDE: https://www.anaconda.com/). If you would like to check this package out without full installation - see section "Tutorial".

1. Install npm: https://docs.npmjs.com/cli/v7/configuring-npm/install
2. Install orca: `npm install -g electron@6.1.4 orca`
3. To create animations, install ImageMagick: https://docs.wand-py.org/en/0.3.5/guide/install.html (on Linux: 'sudo apt-get install libmagickwand-dev')
4. Manually download ``foodwebviz`` package from [GitHub](https://github.com/ibs-pan/foodwebviz) and run the following terminal command from the
top-level source directory (on Windows use e.g. Anaconda Prompt):

    `$ pip install .`


Tutorial
--------
- ``examples/sample_output`` contains examples of visualisations (screenshots of interactive heatmap and graph visualisations)
- ``examples/foodwebviz_tutorial.ipynb`` is an interactive Jupyter Notebook with code examples and functionality overview.

To get information on a specific function/method "function_name" please execute "help(function_name)" in a Jupyter Notebook or Python console.
You can also play with the tutorial notebook without installing the package locally: https://mybinder.org/v2/gh/ibs-pan/foodwebviz/master?filepath=examples%2Fvisualization.ipynb



Testing
-------
To execute the tests run:
$ pytest 

foodwebviz uses the Python ``pytest`` testing package.  You can learn more
about pytest on their [homepage](https://pytest.org).

Bugs \ Questions
-------

In case of questions / problems / bugs please report them [here](https://github.com/ibs-pan/foodwebviz/issues).


Citation
-------

When using foodwebviz package please cite:

Łukasz Pawluczuk, Mateusz Iskrzyński (2021) Food web visualisation: heatmap, interactive graph, animated flow network. https://github.com/ibs-pan/foodwebviz (currently in review in Methods in Ecology and Evolution)

A BibTeX entry for LaTeX users:

@article{foodwebviz,
author={Łukasz Pawluczuk and Mateusz Iskrzyński},
title={Food web visualisation: heatmap, interactive graph, animated flow network},
year={2021},
url={https://github.com/ibs-pan/foodwebviz }
}

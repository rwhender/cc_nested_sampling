# Combined-chain nested sampling

Combined-chain nested sampling is a numerical technique for performing Bayesian model comparison.
It is based on John Skilling's original [nested sampling](http://www.inference.phy.cam.ac.uk/bayesys/) algorithm.
Combined-chain nested sampling adds the ability to split the work of one long nested sampling run over multiple simultaneous shorter runs whose results can be combined afterwards to produce a single result without loss of precision.

This Python 3 module provides an implementation of the technique as described in [Henderson, Goggans, and Cao.
“Combined-chain nested sampling for efficient Bayesian model comparison.”
Digital Signal Processing, volume 70, pages 84–93, 2017.](https://doi.org/10.1016/j.dsp.2017.07.021)
This implementation uses process-based "threading" via the `Pool` class from the `multiprocessing` module to perform multiple simultaneous and independent runs of nested sampling.

## Getting Started

These instructions will get you a copy of the module up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need a Python 3 installation with NumPy and SciPy to run this module.
You'll also want the [dill](https://pypi.python.org/pypi/dill) module if you want to run the example scripts.

### Installing

First, you'll need to get a copy of the source code, either by forking this repository on your system or just downloading the archive.
To install the module, run the setup script,

```
python setup.py install
```


## Running the examples

The example scripts in the `examples` folder can be run to produce results similar to those in the [paper](https://doi.org/10.1016/j.dsp.2017.07.021).
These examples should also give you some insight in how to import and use the classes and functions in the module.
You will need the [dill](https://pypi.python.org/pypi/dill) module to load the data for the `sinusoid_script.py` example.
The script `generate_sinusoids.py` can be used to generate similar data to that in `sinusoid_data_20161004.dill`. 

## License

The [module code](cc_nested_sampling) is licensed under the GNU Lesser General Public License (LGPL), version 3.0.
The [examples](examples) are licensed under the GNU General Public License (GPL), version 3.0.

# OPFython: An Optimum-Path Forest classifier

[![Latest release](https://img.shields.io/github/release/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/releases)
[![License](https://img.shields.io/github/license/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/blob/master/LICENSE)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/issues)

## Welcome to OPFython.
Have you ever wanted to classify data into labels? If yes, OPFython is for you! This package is an innovative way of dealing with an optimum-path forest classifier. From bottom to top, from samples and datasets to the actual classifier, we will foster all research related to this newly trend.

Use OPFython if you need a library or wish to:
* Create your own datasets.
* Design or use pre-loaded state-of-art classifiers.
* Mix-and-match different strategies to solve your problem.
* Because it is cool to classify things.

Read the docs at [opfython.recogna.tech](http://opfython.recogna.tech).

OPFython is compatible with: **Python 2.7-3.6**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy, if you wish to read the code and bump yourself into, just follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with OPFython

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage and follow the example. We have high-level examples for most tasks we could think of.

Or if you wish to learn even more, please take a minute:

OPFython is based on the following structure, and you should pay attention to its tree:

```
- opfython
    - core
        - dataset
        - sample
    - datasets
        - loaded
    - math
        - distribution
        - random
    - utils
        - loader
        - logging
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basic of our structure. They should provide variables and methods that will help to construct other modules. It is composed by the following classes:

```dataset```: A dataset is composed by a number of samples. It will serve as the basis for building OPF's graph.

```sample```: This defines a Sample. Basically, a sample contains a label and a features vector.

### Datasets

Because we need data, right? Datasets are composed by classes and methods that allow to instanciate pre-loaded data. One can see them as a wrapper for loading raw data and creating a Dataset object. 

```loaded```: A loaded dataset already parses the input data from a OPF file and creates an Dataset object.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low level math implementations. From random numbers to distributions generation, you can find your needs on this module.

```distribution```: Package used to handle distributions generation.

```random```: Package used to handle random numbers generation.

### Utils

This is an utilities package. Common things shared across the application should be implemented here. It is better to implement once and use as you wish than re-implementing the same thing over and over again.

```loader```: Module that is responsible for loading files in OPF file format (.csv, .txt or .json).

```logging```: Logging tools to track the progress of the optimization task.

---

## Installation

We belive that everything have to be easy. Not diffucult or daunting, OPFython will be the one-to-go package that you will need, from the very first instalattion to the daily-tasks implementing needs. If you may, just run the following under your most preferende Python environment (raw, conda, virtualenv, whatever)!:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for an additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it's inevitable to acknowlodge that we make mistakes. If you every need to report a bug, report a problem, talk to us, please do so! We will be avaliable at our bests at this repository or recogna@unesp.br.

---

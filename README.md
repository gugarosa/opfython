# OPFython: Python-Inspired Optimum-Path Forest

[![Latest release](https://img.shields.io/github/release/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/issues)
[![License](https://img.shields.io/github/license/gugarosa/opfython.svg)](https://github.com/gugarosa/opfython/blob/master/LICENSE)

## Welcome to OPFython.

Have you ever wanted to classify data into labels? If yes, OPFython is for you! This package is an innovative way of dealing with an optimum-path forest classifier. From bottom to top, from samples and datasets to the actual classifier, we will foster all research related to this new trend.

Use OPFython if you need a library or wish to:
* Create your datasets.
* Design or use pre-loaded state-of-art classifiers.
* Mix-and-match different strategies to solve your problem.
* Because it is cool to classify things.

Read the docs at [opfython.readthedocs.io](https://opfython.readthedocs.io).

OPFython is compatible with: **Python 3.6+** and **PyPy 3.5**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**. Call us.

---

## Getting started: 60 seconds with OPFython

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

OPFython is based on the following structure, and you should pay attention to its tree:

```
- opfython
    - core
        - heap
        - node
        - opf
        - subgraph
    - math
        - distance
        - distribution
        - general
        - random
    - models
        - supervised
    - stream
        - loader
        - parser
        - splitter
    - utils
        - constants
        - converter
        - exception
        - logging
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Math

Just because we are computing stuff, it does not means that we do not need math. Math is the mathematical package, containing low-level math implementations. From random numbers to distributions generation, you can find your needs in this module.

### Models

Each machine learning OPF-based technique is defined in this package. From Supervised OPF to Unsupervised OPF, you can use whatever suits your needs.

### Stream

Every pipeline has its first step, right? The stream package serves as primary methods to load data, parse it into feasible arrays, and split them into the desired sets (training, evaluation, testing).

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing over and over again.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, OPFython will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```Python
pip install opfython
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---

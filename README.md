# Generative quantum eigensolver 

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

GQE is the brand-new algorithm where the training of the generative model mimics the search of the ground state of the Hamiltonian.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- We checked it works with Python 3.8 and 3.10.

### Download
```
git clone --recurse-submodule git@github.com:konakaji/gqe.git
```

### Installation

```
conda create -n gqe python=3.10
```

#### For Linux

```
bash install_linux.sh
```

#### For Mac with M1 chip
```
bash install_mac.sh
```

We did not test for the other environments, so please first try ```bash install_mac.sh```, and report if there are any errors.

## Usage 

```
# At the project root
export PYTHONPATH=`pwd`
cd task
python hydrogen.py
```

## Built With
The following submodules are used:
- https://github.com/konakaji/qwrapper 
- https://github.com/konakaji/qswift

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

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
git submodule update --force --recursive --init --remote # If you fail to fetch the submodules.
```

### Installation

#### Step 1
```
conda create -n gqe python=3.10
```

#### Step 2
- For Linux

```
bash install_linux.sh
```

- For Mac with M1 chip
```
bash install_mac.sh
```

- For the other environment: we did not test for the other environments. Please first try 'bash install_mac.sh', and report if there are any errors.

## Usage 
### First example run
```
# At the project root
export PYTHONPATH=`pwd`
cd task
python hydrogen.py
```

### Others
To be documented.

## Built With
The following submodules are used:
- https://github.com/konakaji/qwrapper 
- https://github.com/konakaji/qswift

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

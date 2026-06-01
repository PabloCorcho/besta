# Bayesian Estimator for Stellar Population Analysis (BESTA)

[![Documentation Status](https://readthedocs.org/projects/besta/badge/?version=latest)](https://besta.readthedocs.io/en/latest/?badge=latest)
[![test](https://github.com/PabloCorcho/besta/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/PabloCorcho/besta/actions/workflows/test.yml)

BESTA is a Python library for Bayesian inference of galaxy stellar population properties from spectroscopic and/or photometric observations.

It provides a practical interface to build fitting workflows on top of:

- [Population Synthesis Toolkit (PST)](https://github.com/paranoya/population-synthesis-toolkit) for stellar population modeling.
- [CosmoSIS](https://cosmosis.readthedocs.io) for parameter estimation and sampling.

Full user and API documentation: [besta.readthedocs.io](https://besta.readthedocs.io)

## What BESTA Can Do

BESTA supports end-to-end stellar population fitting workflows, including:

- Spectroscopic and photometric inference workflows.
- Joint fitting of stellar population and kinematics.
- Configurable star formation history (SFH) parameterizations.
- Grid-based and emulator-based SFH inference options.
- Sequential and batch execution of fitting pipelines.
- Post-processing and visualization utilities for fitted solutions.

For detailed pipeline setup, module-level configuration, and examples, see the online documentation:

- [Quick guide](https://besta.readthedocs.io/en/latest/quick_guide.html)
- [Configuration reference](https://besta.readthedocs.io/en/latest/configuration.html)
- [Pipeline manager docs](https://besta.readthedocs.io/en/latest/manager.html)
- [Tutorials](https://besta.readthedocs.io/en/latest/index.html)

## Installation

### Python version

- `Python >= 3.10`

### Install from PyPI

```bash
pip install besta
```

### Install from source

Depending on your platform and scientific Python installation, you may also need system libraries typically used by MPI and linear algebra backends (for example OpenMPI, BLAS/LAPACK, and a Fortran compiler).

```bash
git clone https://github.com/PabloCorcho/besta.git
cd besta
python -m pip install -r requirements.txt
python -m pip install .
```

### Ubuntu/Debian example:

```bash
sudo apt update
sudo apt install -y \
	gfortran \
	liblapack-dev \
	libopenblas-dev \
	openmpi-bin \
	openmpi-common \
	libopenmpi-dev \
	libgtk2.0-dev
```

Package names can vary by distribution and version. If your environment already provides BLAS/LAPACK and MPI through conda, you may not need to install all system-level packages.

## Contributing

Contributions are welcome, including bug fixes, new features, tests, docs, and tutorials.

### Report issues

- Open issues or feature requests at: [github.com/PabloCorcho/besta/issues](https://github.com/PabloCorcho/besta/issues)

### Contribute code

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes and add or update tests.
4. Update documentation when behavior or APIs change.
5. Open a pull request with a clear summary and motivation.

### Development quick start

```bash
git clone https://github.com/PabloCorcho/besta.git
cd besta
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

## Citation

If BESTA contributes to your research, please cite the project and acknowledge the software in your publication.

## License

BSD 3-Clause.

## Contact

For questions, please send an email to p.corcho.caballero@rug.nl

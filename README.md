# QCompress
QCompress is an implementation of the quantum autoencoder using Forest and OpenFermion.

### Objective
This was our project for Rigetti Computing's first quantum computing hackathon. Our goal was to create a flexible framework for the quantum autoencoder (QAE) that can be used to compress quantum data. This autoencoder implementation is based on the work by [Romero et al](https://arxiv.org/abs/1612.02806).

### Demo
We've included a demonstration of the quantum autoencoder code in `qae_h2_demo.ipynb`, in which we
compress the ground states of molecular hydrogen.

### Dependencies and Versions Used
- Python 3.5
- [pyQuil](https://github.com/rigetticomputing/pyquil) 2.0.0
- [OpenFermion](https://github.com/quantumlib/OpenFermion) 0.6
- [forestopenfermion](https://github.com/rigetticomputing/forestopenfermion) 0.0.3
- [Grove](https://github.com/rigetticomputing/grove) 1.6.0

### Authors
[Sukin Sim (Hannah)](https://github.com/hsim13372), [Evan Anderson](https://github.com/ejdanderson), Eric Brown, Jonathan Romero

### Fixes
We note that there is a lot of room for improvement and fixes. Please feel free to submit pull requests!

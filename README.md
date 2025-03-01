
# SPRING Project -- Integrating symbolic reasoning into neural generative models for design generation


## Introduction
This repository contains the initial release of the Spatial Reasoning Integrated Generator (SPRING) for design generation, as detailed in our accompanying paper. SPRING is a novel approach that integrates neural and symbolic reasoning to produce designs that not only are aesthetically pleasing but also adhere to explicit user specifications and implicit rules of aesthetics, utility, and convenience.

## Current State of the Code
The code available in this repository is a preliminary version and is meant for basic testing and experimentation. It allows users to input a background image and specification, outputting the completed scene. This version is not exhaustive in terms of experiments and extra functionalities. A more complete and readable version of the code will replace it soon.

## Features
- A basic parser for a subset of constraints is implemented.
- The full set of constraints can be accessed by editing lines 121 or 155 in `run.py`, as indicated by the "NOTE: Replace this with desired testing spec" comments.
- Training code is included, and a pretrained model named `core.pt` is available.

## About SPRING
SPRING embeds a neural and symbolic integrated spatial reasoning module within a deep generative network. The spatial reasoning module determines the locations of objects to be generated in the form of bounding boxes. These are predicted by a recursive neural network and filtered by symbolic constraint satisfaction, ensuring the output satisfies user requirements. SPRING provides interpretability by allowing the visualization and diagnosis of the generation process through bounding boxes. It is also capable of handling novel user specifications not present in the training set, excelling in zero-shot transfer of new constraints.


## Usage
Users can test the program using the following command:
```shell
python .\run.py --wd .\logs\ --mode demo --bg_img .\logs\bg.jfif
```
The default specification currently loaded is "A black microwave and a nice looking cooking oven". Both objects are below 200 milles. The microwave is above the oven and either completely to the left or completely to the right of it. The oven is also taller than the microwave.

## Citation
If you find this useful in your work, please cite:
```
@article{Jacobson2025Integrating,
title = {Integrating symbolic reasoning into neural generative models for design generation},
journal = {Artificial Intelligence},
volume = {339},
pages = {104257},
year = {2025},
issn = {0004-3702},
doi = {https://doi.org/10.1016/j.artint.2024.104257},
url = {https://www.sciencedirect.com/science/article/pii/S0004370224001930},
author = {Maxwell J. Jacobson and Yexiang Xue},
keywords = {Constraint reasoning, Neural generative models, Constrained content generation},
}
```

Find the paper at: 
 - https://www.sciencedirect.com/science/article/abs/pii/S0004370224001930
 - https://arxiv.org/abs/2310.09383

## Disclaimer
This is not the final version of the SPRING project. The code will be updated to a more comprehensive version, enhancing readability and functionality.



Thank you for your interest in the SPRING project.

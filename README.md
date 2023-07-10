# An online learning algorithm to play discounted repeated games in wireless networks

## Introduction

CThis code implements the LEWIS (LEarning WIth Security) algorithm and the two use cases published in Parras, J., Apellániz, P.A., & Zazo, S. (2022). An online learning algorithm to play discounted repeated games in wireless networks. Engineering Applications of Artificial Intelligence, 107, 104520. [DOI](https://doi.org/10.1016/j.engappai.2021.104520).

## Launch

To launch each of the use cases, simply run the two python scripts in each folder (`base_stations.py` for the Distributed Power Control, and ìnterference_2.py` for the Interference Mitigation case). 

You will need the following libraries:
* `numpy`
* `matplotlib`
* `cvxopt` (See [here](https://cvxopt.org/) for more details).
* `tikzplotlib` (only if you want to obtain the `.tikz` graphs).

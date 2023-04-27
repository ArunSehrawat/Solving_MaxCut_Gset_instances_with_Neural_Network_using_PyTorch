# Solving Max-Cut Gset instances with Neural Network using PyTorch

# [Max-cut problem][1]

Given an undirected graph $\mathcal{G=(V,E)}$, where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of edges, the max-cut problem asks to partition $\mathcal{V}$ into two disjoint sets, say $\mathcal{S}$ and $\mathcal{T}$, such that the sum of the weights of the edges---called the cut value---of edges between $\mathcal{S}$ and $\mathcal{T}$ maximized. 



$\text{cut value} =\sum_{i<j} w_{ij}\frac{1-z_i z_j}{2} = \frac{1}{2}(\text{total weight} - E)$

$\text{total weight} = \sum_{i<j} w_{ij}$ is a constant.

$E = \text{energy}(\textbf{z}) = \sum_{i<j} w_{ij}z_i z_j$ is an energy of a spin glass (Ising) model.

Maximization of the $\text{cut value}$ is equivalent to minimization of the energy $E$, see [A. Lucas, Ising formulations of many NP problems, Front. Physics 2:5 (2014).][2], 
over the dichotomic variables $z_i = +1,-1$ if the node $i\in \mathcal{S}, \mathcal{T}$, respectively.



The max-cut problem has important applications in various fields, including computer vision, statistical physics, and computational biology. It is also known to be NP-hard, which means that it is computationally difficult to solve optimally for large instances of the problem. Therefore, various approximation algorithms and heuristics have been developed to tackle the problem.


[1]:https://en.wikipedia.org/wiki/Maximum_cut


[2]:https://doi.org/10.3389/fphy.2014.00005

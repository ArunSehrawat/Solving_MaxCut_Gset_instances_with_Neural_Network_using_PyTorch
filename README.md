# Solving [Max-cut problem][1] [Gset instances][3] with Neural Network using PyTorch


Given an undirected graph $\mathcal{G=(V,E)}$, where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of edges, the max-cut problem asks to partition $\mathcal{V}$ into two disjoint sets, say $\mathcal{S}$ and $\mathcal{T}$, such that the sum of the weights of the 
edges---called the cut value---of edges between $\mathcal{S}$ and $\mathcal{T}$ maximized. 

$\text{cut value} =\sum_{i,j} w_{ij}\frac{1-z_i z_j}{2} = \frac{1}{2}(\text{total weight} - E)$

$\text{total weight} = \sum_{i,j} w_{ij}$ is a constant.

$E = \text{energy}(\textbf{z}) = \sum_{i,j} w_{ij}z_i z_j$ is an energy of a spin glass (Ising) model.

Maximization of the $\text{cut value}$ is equivalent to minimization of the energy $E$, see [A. Lucas, Ising formulations of many NP problems, Front. Physics 2:5 (2014).][2], 
over the dichotomic variables $z_i = +1,-1$ if the node $i\in \mathcal{S}, \mathcal{T}$, respectively.



The max-cut problem has important applications in various fields, including computer vision, statistical physics, and computational biology. It is also known to be NP-hard, which means that it is computationally difficult to solve optimally for large instances of the problem. Therefore, various approximation algorithms and heuristics have been developed to tackle the problem.


[1]:https://en.wikipedia.org/wiki/Maximum_cut

[2]:https://doi.org/10.3389/fphy.2014.00005

[3]:https://web.stanford.edu/~yyye/yyye/Gset


## Minimizer Neural Network (MNN)

We are using Multi-Layer Perceptron (MLP) as our Minimizer Neural Network. The MLP takes an $m$-component learnable vector $\textbf{x}$ as input, passes it through $L$ layers with learnable parameters 
$\boldsymbol{\theta}:=\{\theta^{1},\cdots,\theta^{L}\}$, and gives an
$n$-component output vector $\textbf{z}$. Each component of $\textbf{z}$ lies in the interval 
$[-1, 1]$. As a whole, Multi-Layer Perceptron acts as a continuous (differentiable) function 

$f: \mathbb{R}^K \longrightarrow [-1,1]$ such that

$f(\textbf{x},\boldsymbol{\theta})=\textbf{z}$ and $K$ is the total number of learnable parameters.

We feed the output $\textbf{z}$ into the $\text{loss}(\textbf{x},\boldsymbol{\theta})=\text{energy}(\textbf{z})$ and minimize it with Adam optimizer.
After minimization, we got $\textbf{z}_{\text{out}}$, whose components we map to discrete values as

$z \longrightarrow 1$ and $-1$ for $z<0$ and $z\leq0$, respectively

to achieve a max-cut solution (displayed by the graph below).



![MLP_MaxCut.png](attachment:MLP_MaxCut.png)

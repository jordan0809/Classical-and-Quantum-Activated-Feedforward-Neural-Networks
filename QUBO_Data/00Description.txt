Datasets for MaxCut, random QUBO, MWIS, and TSP can be found in this directory.

For MaxCut, Random QUBO, and MWIS, the file name is formatted as `{number of variables}_{problem_class}_instances.data`

The problem class of MaxCut is `maxcut` and random QUBO is `QUBO`.

For instances with fewer than 40 variables, .data file is a list containing 2 elements---a list of QUBO matrices and a list of solution cost values. 

For 80- and 200-variable instances, .data file simply contains a list of QUBO matrices.

For TSP, the file name is {number of cities}_tsp_instances.data

Except for the 14-city TSP, all the .data files contain a list with 3 elements---a list of QUBO matrices, a list of solution cost values, and a list of distance matrices.

The 14-city TSP file contains only the QUBO matrices and the distance matrices.

All the QUBO matrices in the datasets are torch.tensor object. These matrices can be converted to numpy arrays using `.detach().numpy()`. However, both the FNN and QCED solvers take tensors as input. 

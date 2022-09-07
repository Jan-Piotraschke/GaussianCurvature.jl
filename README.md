# GaussianCurvature.jl
Recipes for plotting gaussian curvature

## Goal
Workflow for ...
1. creating gaussian curvatures 3D models with the help of PDEs
2. deform the 3D model slightliy in [Blender](https://www.blender.org)
3. fit the PDEs onto the changed 3D model with Hidden Physics Models


## PINNs as a candidate for a FaustAI (_currently in review; maybe UDEs are better, or the SINDy approach_)

[PINNs](https://maziarraissi.github.io/PINNs/) can be designed to solve two classes of problems:
- data-driven solution
- data-driven discovery  

of partial differential equations.  

Here we implemented the data-driven discovery given noisy and incomplete measurements.  
It is important to understand that the PDEs (that govern a given data-set) get embeded into the learning process of the NN.  
Explicitly speaking, the **PDEs get embeded into the cost function** of the NN.  
With that, the embeded PDEs act as a regularization agent that limits the space of admissible solutions of the NN training.  
The PINN alone does not find any unknown/missing terms of the PDE problem.  
**It only adjusts the unknown PDE parameters** as part of its cost function.

## Package management

The `Project.toml` and `Manifest.toml` contain the package definitions for Julia.  
This allwos us to manage packages with Julias built-in package manager.  
This is how you manage packages with it:

- Open a terminal in the root of the project
- Run `julia`
- type `]` (closing square bracket)
- run `activate .`
- use `add <PackageName>` to add a new package
- use `rm <PackageName>` to remove a package
- use `up <PackageName>` to update a package to a newer version

All your modifications to the packages will be reflected in the `Project.toml` and `Manifest.toml` respectively.

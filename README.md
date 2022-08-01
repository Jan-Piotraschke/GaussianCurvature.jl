# GaussianCurvature.jl
Recipes for plotting gaussian curvature

## Goal
Workflow for ...
1. creating gaussian curvatures 3D models with the help of PDEs
2. deform the 3D model slightliy in [Blender](https://www.blender.org)
3. fit the PDEs onto the changed 3D model with Hidden Physics Models

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

using Meshing  # ok package
using MeshIO  # good package
using FileIO  # good package
using GeometryBasics  # great package
using LinearAlgebra  # great package
import Makie  # great package
using GLMakie  # great package


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONFIGURATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

GLMakie.activate!()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MATH + GEOMETRY WITH GeometryBasics
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

gyroid(v) = cos(v[1])*sin(v[2])+cos(v[2])*sin(v[3])+cos(v[3])*sin(v[1])
gyroid_shell(v) = max(gyroid(v)-0.4,-gyroid(v)-0.4)

# generate directly using GeometryBasics API
# Rect specifies the sampling intervals
gy_mesh = GeometryBasics.Mesh(gyroid_shell, Rect(Vec(0,0,0),Vec(pi*4,pi*4,pi*4)), Meshing.MarchingCubes(), samples=(50,50,50))

# change extension to save as STL, PLY, OBJ, OFF
FileIO.save("assets/gyroid.stl", gy_mesh)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

Makie.mesh(gy_mesh, color=[norm(v) for v in coordinates(gy_mesh)])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# LOAD FILE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
mesh_loaded = load("assets/gyroid_changed.stl")
Makie.mesh(mesh_loaded, color=[norm(v) for v in coordinates(mesh_loaded)])

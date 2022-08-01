# GLMakie is the native, desktop-based backend, and is the most feature-complete.
# It requires an OpenGL enabled graphics card with OpenGL version 3.3 or higher.
# see: https://makie.juliaplots.org/stable/documentation/backends/glmakie/
# Chris von SciML möchte deren ganze Visualisierung über Makie.jl laufen lassen


# using Pkg
# Pkg.add("GLMakie")
# Pkg.add("LinearAlgebra")
# Pkg.add("GeometryBasics")
# Pkg.add("Makie")

using GLMakie, LinearAlgebra, GeometryBasics
using Makie


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CONFIGURATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Start the OpenGL engine !
GLMakie.activate!()
GLMakie.set_window_config!(
    framerate = 10,
    title = "Arrows on a surface"
)

n = 20


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MATH + GEOMETRY WITH GeometryBasics
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ]
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ]
z = [cospi(θ) for θ in θ, φ in φ]

# ! generate directly using GeometryBasics API
f(x,y,z) = x*exp(cos(y)*z)
∇f(x,y,z) = Point3f0(exp(cos(y)*z), -sin(y)*z*x*exp(cos(y)*z), x*cos(y)*exp(cos(y)*z))  # 'Point3f0' from GeometryBasics
∇ˢf(x,y,z) = ∇f(x,y,z) - Point3f0(x,y,z)*dot(Point3f0(x,y,z), ∇f(x,y,z))

# calculate the properties of all the arrows:
arrow_startpoints = vec(Point3f0.(x, y, z))
arrow_directions = vec(∇ˢf.(x, y, z)) .* 0.1f0


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

scene = Makie.Scene(resolution = (400,400));

Makie.surface(x, y, z)   # fig, ax, pltobj = surface(x, y, z)
arr = Makie.arrows!(
    arrow_startpoints, arrow_directions,
    arrowsize = 0.05, linecolor = (:white, 0.7), linewidth = 0.02, lengthscale = 0.1
)

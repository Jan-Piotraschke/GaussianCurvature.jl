using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using ModelingToolkit
using IntervalSets
using Makie
using GLMakie
using MeshIO  # good package
using FileIO  # good package

GLMakie.activate!()


"""
    construct_NN()

Construct the NN
number of dimensions = length of the domains
"""
function construct_NN(dim)
      chain = Lux.Chain(Dense(dim,16,Lux.σ),Dense(16,16,Lux.σ),Dense(16,1))

      return chain
end


"""
    solve_pde_using_pinn(_prob)

Solve the PDE system using a designed PINN
"""
function solve_pde_using_pinn(_prob)
      # Optimizer
      opt = OptimizationOptimJL.BFGS()

      # Callback function
      callback = function (p,l)
            println("Current loss is: $l")
            return false
      end

      return Optimization.solve(_prob, opt, callback = callback, maxiters=1000)
end


# '@' indicates that creation of a 'macro'
# macros are part of metaprogramming
@parameters x y  # unsere Variablen, die die Ableitung formen
@variables u(..)
Dxx = Symbolics.Differential(x)^2
Dyy = Symbolics.Differential(y)^2

# 2D PDE: Poisson equation
# ! '~' indicates that it is an equation
eqs = [Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)]  # TODO: pass this eq to LaTeX and print it next to the plot

# Boundary conditions
bcs = [u(0,y) ~ 0.0,
       u(1,y) ~ 0.0,
       u(x,0) ~ 0.0,
       u(x,1) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]

# define the PDE system
# man setzt sozusgaen obige Parameter und Variaben ein und weißt denen ihre jeweilige Bedeutung damit zu
@named pde_system =  ModelingToolkit.PDESystem(eqs,bcs,domains,[x,y],[u(x, y)])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# construct the NN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

chain = construct_NN(length(domains))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# build PINN algorithm
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dx = 0.05
strategy = GridTraining(dx)

# transforms a PDESystem into an OptimizationProblem using the Physics-Informed Neural Networks (PINN) methodology
# !param_estim: whether the parameters of the PDE should be included in the values sent to the additional_loss function.
# uses a DiffEqFlux.jl neural network to solve the differential equation
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, param_estim = false)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# transform the PDE system into a PINNs problem using the discretize method
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

prob = SciMLBase.discretize(pde_system, discretization)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SOLVE THE PDE USING PINNs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

res = solve_pde_using_pinn(prob)
phi = discretization.phi


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# We can plot the predicted solution of the PDE and compare it with the analytical solution in order to 
# plot the relative error.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

xs,ys = [IntervalSets.infimum(d.domain):dx/10:IntervalSets.supremum(d.domain) for d in domains]
u_predict = reshape([first(phi([x,y],res.u)) for x in xs for y in ys],(length(xs),length(ys)))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # 2D Code
# f = Figure()
# Axis(f[1, 1])
# co = contourf!(xs, ys, u_predict)
# Colorbar(f[1, 2], co)

# # 3D Code
# scene = Makie.Scene(resolution = (400,400));
# surface = Makie.surface(xs, ys, u_predict, axis=(type=Axis3,))

# TODO: make this a "Face-Vertex Mesh" -> the most widely used mesh representation
# NOTE: 'Face': usually consists of triangles
using Meshes, GeometryBasics

faces = decompose(QuadFace{GLIndex}, Tesselation(Rect(0, 0, 1, 1), size(u_predict)))   # ? faces of the shape of a Quadrat
# faces = decompose(UV(GeometryBasics.Vec3f), Tesselation(Rect(0, 0, 1, 1), size(u_predict)))
gl_points = GLMakie.Point.(xs,ys,u_predict) |> vec

gb_mesh = GeometryBasics.Mesh(gl_points, faces)  # create the mesh
FileIO.save("assets/poisson.stl", gb_mesh)

f, ax, pl = Makie.mesh(gb_mesh,  color = rand(100, 100), colormap=:blues)  # plot the mesh


# GeometryBasics.Mesh(Tesselation(surface,64))
# gb_rect = GeometryBasics.Rect3f(GeometryBasics.Vec3f(xs),GeometryBasics.Vec3f(xs))


# GeometryBasics.connect.([tuple(xs'),tuple(ys')], Quadrangle)







#######
using GeometryBasics, LinearAlgebra, GLMakie, FileIO

# Create vertices for a Sphere
r = 0.5f0
n = 30
θ = LinRange(0, pi, n)
φ2 = LinRange(0, 2pi, 2 * n)
x2 = [r * cos(φv) * sin(θv) for θv in θ, φv in φ2]
y2 = [r * sin(φv) * sin(θv) for θv in θ, φv in φ2]
z2 = [r * cos(θv) for θv in θ, φv in 2φ2]
points = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(x2, y2, z2)])

# The coordinates form a matrix, so to connect neighboring vertices with a face
# we can just use the faces of a rectangle with the same dimension as the matrix:
faces = decompose(QuadFace{GLIndex}, Tesselation(Rect(0, 0, 1, 1), size(z2)))
# Normals of a centered sphere are easy, they're just the vertices normalized.
normals = normalize.(points)

# Now we generate UV coordinates, which map the image (texture) to the vertices.
# (0, 0) means lower left edge of the image, while (1, 1) means upper right corner.
function gen_uv(shift)
    return vec(map(CartesianIndices(size(z2))) do ci
        tup = ((ci[1], ci[2]) .- 1) ./ ((size(z2) .* shift) .- 1)
        return Vec2f(reverse(tup))
    end)
end

# We add some shift to demonstrate how UVs work:
uv = gen_uv(0.0)
# We can use a Buffer to update single elements in an array directly on the GPU
# with GLMakie. They work just like normal arrays, but forward any updates written to them directly to the GPU
uv_buff = Buffer(uv)
gb_mesh = GeometryBasics.Mesh(meta(points; uv=uv_buff, normals), faces)

f, ax, pl = mesh(gb_mesh,  color = rand(100, 100), colormap=:blues)
wireframe!(ax, gb_mesh, color=(:black, 0.2), linewidth=2, transparency=true)
record(f, "uv_mesh.mp4", LinRange(0, 1, 100)) do shift
    uv_buff[1:end] = gen_uv(shift)
end
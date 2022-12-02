using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using ModelingToolkit
using IntervalSets
using Makie  # great package
using GLMakie  # great package
using MeshIO  # good package
using FileIO  # good package
using Meshes
using GeometryBasics  # great package

GLMakie.activate!()


# TODO: maybe implement 'dropout' logic into the NN which avoids overfitting by randomly turning off some neurons
"""
    construct_NN()

Aufgabe: wir möchten stetige Werte vorhersagen -> Regressionsproblem

Construct the NN layers Regression möchten wir stetige Werte vorhersagen
NN consists out of 3 dense layers: every neuron in the layer n is connected to all the neurons in the n-1 layer
activation function of the neurons in the i th layer: sigmoid (=σ) activation function

input of the first layer: number of dimensions = length of the domains
output of the last layer: 1 -> neccessary for regression ML
"""
function construct_NN(dim)
    n = 20
    chain = Lux.Chain(Dense(dim,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1))
    # ? NOTE: when should we construct more chains for the PINN
    return chain
end


"""
    solve_pde_using_pinn(_prob)

Solve the PDE system using a designed PINN
Network learning is simply minimizing a cost function (e.g. with gradient descent methode)!
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


"""
    combine_diffeq_sols_into_mesh(xs, ys, diffeq_sol)

This is a "Face-Vertex Mesh" -> the most widely used mesh representation
'Face': usually consists of triangles
"""
function combine_diffeq_sols_into_mesh(xs, ys, diffeq_sol::Array)
    faces = decompose(GLTriangleFace, Tesselation(Rect(0, 0, 1, 1), size(u_predict)))

    xs_vec = xs' .* ones(length(xs)) |> vec  # create meshgrid
    ys_vec = ones(length(ys))' .* ys |> vec  # create meshgrid
    diffeq_sol_vec = diffeq_sol |> vec

    gl_points = GLMakie.Point.(xs_vec,ys_vec,diffeq_sol_vec) |> vec

    gb_mesh = GeometryBasics.Mesh(gl_points, faces)  # create the mesh

    return gb_mesh
end


@parameters t, x 
@variables u1(..), u2(..), u3(..) 
Dt=Differential(t) 
Dtt =Differential(t)^2
Dx=Differential(x) 
Dxx =Differential(x)^2 

# differential-algebraic equation system
eqs = [Dtt(u1(t,x))~ Dxx(u1(t,x)) + u3(t,x)*sin(pi*x),
     Dtt(u2(t,x)) ~ Dxx(u2(t,x)) + u3(t,x)*cos(pi*x), 
     0. ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x) - exp( -t)] 
     
bcs = [u1(0,x) ~ sin(pi*x), 
        u2(0,x) ~ cos(pi*x), 
        Dt(u1(0,x)) ~ -sin(pi*x),
        Dt(u2(0,x)) ~ -cos(pi*x), 
        u1(t,0) ~ 0., 
        u2(t,0) ~ exp(-t), 
        u1(t,1) ~ 0., 
        u2(t,1) ~ -exp(-t)]

# Space and time domains 
domains = [t ∈ IntervalDomain(0.0,1.0),
            x ∈ IntervalDomain(0.0,1.0)] 

@named pde_system = PDESystem(eqs,bcs,domains,[t,x],[u1(t,x),u2(t,x),u3(t,x)])



# NOTE: the construction and visualization of the NN seems to be always the same.


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# construct the NN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

chain = construct_NN(length(domains))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# build PINN algorithm
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dx = 0.2
strategy = GridTraining(dx)

# transforms a PDESystem into an OptimizationProblem using the Physics-Informed Neural Networks (PINN) methodology
# ! param_estim: whether the parameters of the PDE should be included in the values sent to the additional_loss function.
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
# transform your data into a mesh and save it
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

gb_mesh = combine_diffeq_sols_into_mesh(xs, ys, u_predict)
FileIO.save("assets/pdae.stl", gb_mesh)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 3D Code
scene = Makie.Scene(resolution = (400,400));
f, ax, pl = Makie.mesh(gb_mesh, axis=(type=Axis3,))  # plot the mesh
# wireframe!(ax, gb_mesh, color=(:black, 0.2), linewidth=2, transparency=true)  # only for the asthetic

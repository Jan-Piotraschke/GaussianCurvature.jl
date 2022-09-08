using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval
using ModelingToolkit
using IntervalSets
using Makie
using GLMakie

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

# 2D Code
#f = Figure()
#Axis(f[1, 1])
#co = contourf!(xs, ys, u_predict)
#Colorbar(f[1, 2], co)

scene = Makie.Scene(resolution = (400,400));
Makie.surface(xs, ys, u_predict, axis=(type=Axis3,))

# TODO: export the plot to a mesh and then to a '.stl' file

using ModelingToolkit, OrdinaryDiffEq
using DataFrames
using Makie
using GLMakie

GLMakie.activate!()


"""
Transform your ODE solution into a DataFrame
"""
function create_dataframe(sol, sys::ODESystem)
      _data_sol = DataFrame(reduce(hcat, sol.u)', :auto)

      # ! Julia has Perl-compatible regular expressions
      col_names = match(r"\[(.*?)\]", string(sys.states)).captures[1]
      col_names = split(col_names, ", ")

      rename!(_data_sol, col_names)

      return _data_sol
end


# '@' indicates that creation of a 'macro'
# macros are part of metaprogramming
@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Symbolics.Differential(t)

eqs = [D(D(x)) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

@named sys = ODESystem(eqs)
sys = ode_order_lowering(sys)

u0 = [D(x) => 2.0,
      x => 1.0,
      y => 0.0,
      z => 0.0]

p  = [σ => 28.0,
      ρ => 10.0,
      β => 8/3]

tspan = (0.0,100.0)
prob = ODEProblem(sys,u0,tspan,p,jac=true)
sol = solve(prob,Tsit5())

data_sol = create_dataframe(sol, sys)
x = select(data_sol, "x(t)") |> Matrix |> vec
y = select(data_sol, "y(t)") |> Matrix |> vec


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PLOTTING DONE BY MAKIE.jl
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

Makie.lines(x, y, color = :blue)







########################### another code -> solve Poisson equation

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
"""
function construct_NN()
      dim = 2  # number of dimensions
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
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)  # TODO: pass this eq to LaTeX and print it next to the plot

# Boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ 0.0,
       u(x,0) ~ 0.0, u(x,1) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# construct the NN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

chain = construct_NN()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# build PINN algorithm
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dx = 0.05
# transforms a PDESystem into an OptimizationProblem using the Physics-Informed Neural Networks (PINN) methodology
# !param_estim: whether the parameters of the PDE should be included in the values sent to the additional_loss function.
discretization = NeuralPDE.PhysicsInformedNN(chain, GridTraining(dx), param_estim = false)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# define the PDESystem and create PINNs problem using the discretize method
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@named pde_system =  ModelingToolkit.PDESystem(eq,bcs,domains,[x,y],[u(x, y)])
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

f = Figure()
Axis(f[1, 1])
co = contourf!(xs, ys, u_predict)
Colorbar(f[1, 2], co)

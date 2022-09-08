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

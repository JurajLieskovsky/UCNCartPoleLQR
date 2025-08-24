using Revise

using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using DiffEqCallbacks, NonlinearSolve
using Plots
using MatrixEquations

using UCNCartPoleODE
# using MeshCatBenchmarkMechanisms

cartpole = UCNCartPoleODE.Model(9.81, 1, 0.1, 0.2)

# Equlibrium
s_eq = 0
q_eq = [0, 1.0]
v_eq = 0
ω_eq = 0

x_eq = vcat(s_eq, q_eq, v_eq, ω_eq)
u_eq = zeros(UCNCartPoleODE.nu)

# Control period
h = 1e-2

# Linearization
"""Euler integration with zero-order hold on u"""
dt_dynamics(x, u) = x + h * UCNCartPoleODE.f(cartpole, x, u)

fx = ForwardDiff.jacobian(x_ -> dt_dynamics(x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> dt_dynamics(x_eq, u_), u_eq)

E = UCNCartPoleODE.jacobian(x_eq)
A = E' * fx * E
B = E' * fu

# LQR design
Q = diagm([1e1, 1e2, 1, 1])
R = Matrix(I(1))

S, _ = MatrixEquations.ared(A, B, R, Q)
K = inv(R + B' * S * B) * B' * S * A

controller(x) =  u_eq - K * UCNCartPoleODE.state_difference(x, x_eq)

# Simulation
tspan = (0.0, 5.0)
θ = 7 * pi / 8
x0 = vcat(s_eq, [cos(θ / 2), sin(θ / 2)], v_eq, ω_eq)

## Callbacks
ControllerCallback = PeriodicCallback(i -> i.p .= controller(i.u), h, initial_affect=true)

saved_values = SavedValues(Float64, Vector{Float64})
InputSavingCallback = SavingCallback((u, t, integrator) -> copy(integrator.p), saved_values)

UCNProjectionCallback = DiffEqCallbacks.ManifoldProjection(
    (x, _, _) -> [x[2:3]' * x[2:3] - 1], autodiff=AutoForwardDiff()
)

cbs = CallbackSet(ControllerCallback, InputSavingCallback, UCNProjectionCallback)

## Problem
prob = ODEProblem(
    (x, p, _) -> UCNCartPoleODE.f(cartpole, x, p),
    x0,
    tspan,
    similar(u_eq)
)

## Solution
sol = solve(prob, callback=cbs)

# Plotting
Δt = 1e-2
ts = tspan[1]:Δt:tspan[2]
xs = map(t -> sol(t), ts)

state_labels = ["s" "q₀" "q₁" "v" "ω"]
input_labels = ["u₀"]

plt = plot(layout=(2, 1))
plot!(
    plt, ts, mapreduce(x -> x[1:3]', vcat, xs),
    label=state_labels, subplot=1
)
plot!(
    plt, saved_values.t, mapreduce(u -> u', vcat, saved_values.saveval),
    label=input_labels, seriestype=:steppost, subplot=2
)

display(plt)

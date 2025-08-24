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

# Linearization
fx = ForwardDiff.jacobian(x_ -> UCNCartPoleODE.f(cartpole, x_, u_eq), x_eq)
fu = ForwardDiff.jacobian(u_ -> UCNCartPoleODE.f(cartpole, x_eq, u_), u_eq)

E = UCNCartPoleODE.jacobian(x_eq)

A = E' * fx * E
B = E' * fu

# LQR design
Q = diagm([1e1, 1e1, 1, 1])
R = Matrix(I(1))

S, _ = MatrixEquations.arec(A, B, R, Q)
K = inv(R) * B' * S

controller(x) = u_eq - K * UCNCartPoleODE.state_difference(x, x_eq)

# Simulation
tspan = (0.0, 5.0)
θ = 7 * pi / 8
x0 = vcat(s_eq, [cos(θ / 2), sin(θ / 2)], v_eq, ω_eq)

UCNProjectionCallback = DiffEqCallbacks.ManifoldProjection(
    (x, _, _) -> [x[2:3]' * x[2:3] - 1], autodiff=AutoForwardDiff()
)

prob = ODEProblem(
    (x, _, _) -> UCNCartPoleODE.f(cartpole, x, controller(x)),
    x0,
    tspan
)
sol = solve(prob, callback=UCNProjectionCallback)

# Plotting
Δt = 1e-2
ts = tspan[1]:Δt:tspan[2]
xs = map(t -> sol(t), ts)
us = map(x -> controller(x), xs)

state_labels = ["s" "q₀" "q₁" "v" "ω"]
input_labels = ["u₀"]

plt = plot(layout=(2, 1))
plot!(plt, ts, mapreduce(x -> x[1:3]', vcat, xs), label=state_labels, subplot=1)
plot!(plt, ts, mapreduce(u -> u', vcat, us), label=input_labels, subplot=2)

display(plt)

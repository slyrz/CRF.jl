using CRF
using Optim

include("util.jl")
include("features.jl")

# Load weather data
X, Y = load(joinpath(@__DIR__, "weather.csv"))

# Use first 5 sequences for parameter estimation
crfs = Sequence[ ]
for i = 1:5
    push!(crfs, Sequence(X[i], Y[i], features))
end

# Since we are looking for parameters maximizing the loglikelihood function
# and Optim.optimize performs function minimization, we negate the
# loglikelihood and it's derivative.
function f(x::Vector)
    -loglikelihood(crfs, Θ=x)
end

function g!(x::Vector, storage::Vector)
    storage .= -loglikelihood_gradient(crfs, Θ=x)
end

l_bfgs = LBFGS()
result = optimize(f, g!, crfs[1].Θ, l_bfgs,
                  Optim.Options(iterations=15, show_trace=true))

println("Minimum: ", result.f_minimum)
println("Parameters: ", result.minimum)
println("f calls: ", result.f_calls)
println("g calls: ", result.g_calls)

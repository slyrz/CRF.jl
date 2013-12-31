using Base
using Base.Test
using CRF

import CRF.Sequence

function feature_vector(yt::Bool, x::Array{Array{Float64,1},1}, t::Int32)
    feature_vector(false, yt, x, t)
end

function feature_vector(yp::Bool, yt::Bool, x::Array{Array{Float64,1},1}, t::Int32)
    v = Features(4+20+20)

    append!(v, [true, false], [true, false]) do z
        ((yp == z[1]) & (yt == z[2]))
    end

    append!(v, [true, false], 0.0:0.1:0.9) do z
        ((yt == z[1]) & (z[2] <= x[t][1] < (z[2] + 0.1)))
    end

    append!(v, [true, false], 0.0:0.1:0.9) do z
        ((yt == z[1]) & (z[2] <= x[t][2] < (z[2] + 0.1)))
    end

    return convert(Array{Float64,1}, v)
end

X = [ rand(Float64, 2) for i = 1:100 ]
Y = [ (x[1] < x[2] < 0.5) for x in X ]

s = Sequence(X, Y, feature_vector)

@test s.x == X
@test s.y == Y
@test s.Y == unique(Y)
@test s.f == feature_vector

@test sum(s.F) > 0
@test loglikelihood(s) < 0.0

@test loglikelihood(s) == loglikelihood(s)
@test loglikelihood(s) != loglikelihood(s, Θ=rand(s.k))

@test size(loglikelihood_gradient(s)) == size(s.F)
@test size(label(s)) == size(s.y)

for l in label(s)
    @test l in (true, false)
end

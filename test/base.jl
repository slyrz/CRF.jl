using Base
using Base.Test
using CRF

import CRF.Sequence

function feature_vector(yt::Bool, x::Array{Array{Float64,1},1}, t::Int32)
    feature_vector(false, yt, x, t)
end

function feature_vector(yp::Bool, yt::Bool, x::Array{Array{Float64,1},1}, t::Int32)
    v = Features(4+20+20)

    for ypv in [true, false], ytv in [true, false]
        @append! v ((yp ==ypv) & (yt == ytv))
    end

    for ytv in [true, false], range = 0.0:0.1:0.9
        @append! v ((yt == ytv) & (range <= x[t][1] < (range + 0.1)))
        @append! v ((yt == ytv) & (range <= x[t][2] < (range + 0.1)))
    end

    @test sum(v.x) != 0
    @test v.i == length(v.x) + 1

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

s = Sequence(X, feature_vector; labels=unique(Y))
@test s.x == X
@test s.Y == unique(Y)
@test isempty(s.y)
@test sum(s.F) == 0


# Unlabeld data without label alphabet shouldn't work
@test_throws s = Sequence(X, feature_vector)

# Observation / label mismatch shouldn't work
@test_throws s = Sequence(X[2:end], Y, feature_vector)

# Feature / weight mismatch shouldn't work
@test_throws s = Sequence(X, Y, feature_vector; Θ=rand(2))

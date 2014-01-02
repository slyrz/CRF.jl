import Base.ArgumentError

type Sequence{Tx,Ty}
    x::Array{Tx,1} # observation sequence
    y::Array{Ty,1} # label sequence
    Y::Array{Ty,1} # alphabet from which labels are drawn

    n::Int32 # length of training data
    d::Int32 # length of label alphabet
    k::Int32 # number of features

    Z::Float64 # normalization constant
    σ::Float64 # regularization constant

    M::Array{Array{Float64, 2}, 1} # transformation matrix
    α::Array{Array{Float64, 1}, 1} # forward vectors
    β::Array{Array{Float64, 1}, 1} # backward vectors

    Θ::Array{Float64, 1} # parameters

    f::Function # function returning feature vector
    F::Array{Float64, 1} # sum of feature vectors 1:n

    function Sequence(x, y, f::Function; Θ=[], labels=[], σ=50.0)
        # If x are unlabeled observations, the labels keyword must provide the
        # alphabet from which labels are drawn.
        if isempty(y)
            if isempty(labels)
                throw(ArgumentError("A label alphabet must be specified for unlabeled observations."))
            end
        else
            if length(x) != length(y)
                throw(ArgumentError("Labels don't match observation length."))
            end
        end

        res = new()
        res.x = x
        res.y = y

        if isempty(labels)
            res.Y = unique(y)
        else
            res.Y = labels
        end

        res.f = f
        res.σ = σ
        res.Z = 0.0

        res.n, = size(res.x)
        res.d, = size(res.Y)
        res.k, = size(res.f(res.Y[1], res.Y[1], res.x, 1))

        res.M = [ zeros(res.d, res.d) for t = 1:res.n ]
        res.α = [ zeros(res.d) for t = 1:res.n ]
        res.β = [ zeros(res.d) for t = 1:res.n ]

        if isempty(Θ)
            Θ = rand(res.k)
        else
            if length(Θ) != res.k
                throw(ArgumentError("Length of Θ doesn't match feature length."))
            end
        end

        res.Θ = Θ
        res.F = zeros(res.k)

        update(res)

        return res
    end
end

Sequence{Tx,Ty}(x::Array{Tx,1}, y::Array{Ty,1}, f::Function; args...) = Sequence{Tx,Ty}(x, y, f; args...)
Sequence{Tx,Ty}(x::Array{Tx,1}, f::Function; labels::Array{Ty,1}=Ty[], args...) = Sequence{Tx,Ty}(x, Ty[], f; labels=labels, args...)

function update(crf::Sequence; Θ=None)
    if (Θ != None)
        if (crf.Θ == Θ)
            return
        end
        crf.Θ[:] = Θ[:]
    end

    # Transition Matrix
    for (t, mt) = enumerate(crf.M)
        for (i, yi) = enumerate(crf.Y), (j, yj) = enumerate(crf.Y)
            mt[i,j] = dot(crf.Θ, crf.f(yi, yj, crf.x, t))
        end
    end

    # Forward Vectors
    for t = 1:crf.n
        for (i, yi) = enumerate(crf.Y)
            if t == 1
                crf.α[t][i] = dot(crf.Θ, crf.f(yi, crf.x, t))
            else
                crf.α[t][i] = logsumexp(crf.M[t][:,i][:] + crf.α[t-1])
            end
        end
    end

    # Backward Vectors
    for t = reverse(1:crf.n)
        for (i, yi) = enumerate(crf.Y)
            if t == crf.n
                crf.β[t][i] = 0.0
            else
                crf.β[t][i] = logsumexp(crf.M[t+1][i,:][:] + crf.β[t+1])
            end
        end
    end

    # Constant normalization factor
    crf.Z = logsumexp(crf.α[end])

    if !isempty(crf.y)
        # Global Feature Vector
        crf.F[:] = 0.0
        for t = 1:crf.n
            if t == 1
                v = crf.f(crf.y[t], crf.x, t)
            else
                v = crf.f(crf.y[t-1], crf.y[t], crf.x, t)
            end
            for i = 1:crf.k
                crf.F[i] += v[i]
            end
        end
    end
end

function update{Tx,Ty}(crfs::Array{Sequence{Tx,Ty},1}; Θ=None)
    if (Θ == None)
        Θ = copy(crfs[1].Θ)
    end

    for crf in crfs
        update(crf, Θ=Θ)
    end
end

function loglikelihood(crf::Sequence; Θ=None)
    if (Θ != None)
        update(crf, Θ=Θ)
    end

    return dot(crf.Θ, crf.F) - crf.Z - dot(crf.Θ, crf.Θ) / (crf.σ ^ 2)
end

function loglikelihood{Tx,Ty}(crfs::Array{Sequence{Tx,Ty},1}; Θ=None)
    sum([ loglikelihood(crf, Θ=Θ) for crf in crfs ])
end

function loglikelihood_gradient(crf::Sequence; Θ=None)
    if (Θ != None) && (crf.Θ != Θ)
        update(crf, Θ=Θ)
    end

    F = copy(crf.F)
    t = 1
    for (i, yi) = enumerate(crf.Y)
        v = crf.f(yi, crf.x, t)
        p = exp(dot(crf.Θ, v) + crf.β[t][i] - crf.Z)

        for u = 1:crf.k
            F[u] -= v[u] * p
        end
    end
    for t = 2:crf.n
        for (i, yi) = enumerate(crf.Y), (j, yj) = enumerate(crf.Y)
            v = crf.f(yi, yj, crf.x, t)
            p = exp(crf.α[t-1][i] + crf.M[t][i,j] + crf.β[t][j] - crf.Z)

            for u = 1:crf.k
                F[u] -= v[u] * p
            end
        end
    end
    return F - (sum(crf.Θ) / crf.σ)
end

function loglikelihood_gradient{Tx,Ty}(crfs::Array{Sequence{Tx,Ty},1}; Θ=None)
    sum([ loglikelihood_gradient(crf, Θ=Θ) for crf in crfs ])
end

function label(crf::Sequence)
    d = zeros(crf.d)
    Q = zeros(crf.d, crf.d)

    M = Any[]
    for t in 2:crf.n
        for i = 1:crf.d, j = 1:crf.d
            Q[i,j] = crf.M[t][i,j] + d[j]
        end
        m = zeros(Int32, crf.d)
        for i = 1:crf.d
            j = indmax(Q[i,:])
            m[i] = j
            d[i] = Q[i,j]
        end
        push!(M, m)
    end

    result = [ indmax(d) ]
    for p in reverse(M)
        push!(result, p[result[end]])
    end
    return [ crf.Y[i] for i in reverse(result) ]
end

function label{Tx,Ty}(crfs::Array{Sequence{Tx,Ty},1})
    [ label(crf) for crf in crfs ]
end

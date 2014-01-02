import Base.start,
       Base.next,
       Base.done,
       Base.ArgumentError

function logsumexp{I<:Number}(x::AbstractArray{I})
    m = maximum(x)
    r = 0.0
    for i = 1:endof(x)
        r += exp(x[i] - m)
    end
    return log(r) + m
end

# Iterator similar to Python's itertools.product. Low-memory-footprint since
# it doesn't keep a list of all iterator combinations in memory.
immutable Product{I<:Tuple}
    itr::I
end

product(itr) = Product(itr)
product(itr...) = Product(itr)

type ProductState
    val::Array{Array{Any,1},1} # values of each iterator ( 1:3 => [1:3] )
    max::Array{Int32,1} # length of each iterator ( 1:3 => 3 )
    idx::Array{Int32,1} # current index configuration
    sta::Array{Any,1} # current value configuration
    yld::Array{Any,1} # values to yield
    p::Int32 # position of current iterator

    function ProductState(p::Product)
        res = new()
        res.val =[ [i] for i in p.itr ]
        for (i, v) in enumerate(res.val)
            if length(v) == 0
                throw(ArgumentError("Empty iterator at index $i."))
            end
        end
        res.max = [ length(v) for v in res.val ]
        res.idx = [ 1 for v in res.val ]
        res.sta = [ v[1] for v in res.val ]
        res.yld = [ v[1] for v in res.val ]
        res.p = 1
        return res
    end
end

function start(p::Product)
    ProductState(p)
end

function next(p::Product, s::ProductState)
    fin = false
    while !fin
        if s.idx[s.p] >= s.max[s.p]
            if s.p > 1 && s.idx[1:(s.p-1)] == s.max[1:(s.p-1)]
                s.idx[1:(s.p-1)] = 1
                for i = 1:(s.p-1)
                    s.idx[i] = 1
                    s.sta[i] = s.val[i][1]
                end
            end
            if s.p == 1
                s.yld[:] = s.sta[:]
                fin = true
            end
            s.idx[s.p] = 1
            s.sta[s.p] = s.val[s.p][1]
            s.p += 1
        else
            if s.p == 1
                s.yld[:] = s.sta[:]
                fin = true
            end
            s.idx[s.p] += 1
            s.sta[s.p] = s.val[s.p][s.idx[s.p]]
            s.p = 1
        end
    end
    return s.yld, s
end

function done(z::Product, s::ProductState)
    (s.p > 1) && (s.idx[1] == 1) && (s.idx[2:end] == s.max[2:end])
end

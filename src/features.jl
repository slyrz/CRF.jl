import Base.append!,
       Base.empty!,
       Base.size,
       Base.convert

type Features
    x::Array{Float64,1}
    i::Int32
    function Features(l::Int32)
        new(zeros(Float64, l), 1)
    end
end

function append!(f::Function,v::Features,itr...)
    for x in product(itr)
        v.x[v.i] = f(x) ; v.i += 1
    end
end

function empty!(v::Features)
    v.i = 1 ; v.x[:] = 0
end

function size(v::Features)
    length(v.x)
end

function convert(::Type{Array{Float64,1}}, v::Features)
    v.x
end

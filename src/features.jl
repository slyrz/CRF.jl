import Base.empty!,
       Base.size,
       Base.convert

mutable struct Features
    x::Array{Float64,1}
    i::Int
    function Features(l::Int)
        new(zeros(Float64, l), 1)
    end
end

macro append!(v,w)
    quote
        $(esc(v)).x[$(esc(v)).i] = float($(esc(w)))
        $(esc(v)).i += 1
    end
end

function empty!(v::Features)
    v.i = 1 ; v.x .= 0
end

function size(v::Features)
    size(v.x)
end

function convert(::Type{Array{Float64,1}}, v::Features)
    v.x
end

function logsumexp(x::AbstractArray{I}) where {I<:Number}
    m = maximum(x)
    r = sum(i -> exp(i - m), x)
    return log(r) + m
end

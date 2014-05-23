function logsumexp{I<:Number}(x::AbstractArray{I})
    m = maximum(x)
    r = 0.0
    for i = 1:endof(x)
        r += exp(x[i] - m)
    end
    return log(r) + m
end

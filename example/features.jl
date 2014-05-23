typealias XT Array{Float64,1}
typealias YT ASCIIString

const labels = YT[ "sunny", "rainy", "foggy" ]

function features(yt::YT, x::AbstractArray{XT}, t::Int)
    features("", yt, x, t)
end

function features(yp::YT, yt::YT, x::AbstractArray{XT}, t::Int)
    res = Features(6+9+36+33+84)

    for yval in labels
        @append! res (yp == yval)
        @append! res (yt == yval)
    end

    for ypval in labels, ytval in labels
        @append! res ((yp == ypval) & (yt == ytval))
    end

    for ypval in labels, ytval in labels
        @append! res ((yp == ypval) & (yt == ytval) & (x[t][1] > 0.0))
        @append! res ((yp == ypval) & (yt == ytval) & (x[t][2] > 0.0))
        @append! res ((yp == ypval) & (yt == ytval) & (x[t][1] < 0.0))
        @append! res ((yp == ypval) & (yt == ytval) & (x[t][2] < 0.0))
    end

    for ytval in labels, diff = 0.0:10.0
        @append! res ((yt == ytval) & (diff <= abs(x[t][1] - x[t][2]) < (diff + 1.0)))
    end

    for yval in labels, tres = 0.0:13.0
        @append! res ((yt == yval) & (tres <= abs(x[t][1]) < (tres + 1.0)))
        @append! res ((yt == yval) & (tres <= abs(x[t][2]) < (tres + 1.0)))
    end

    return convert(Array{Float64,1}, res)
end

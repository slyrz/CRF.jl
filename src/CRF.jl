module CRF
export
    Sequence,
    Features,
    @append!,
    empty!,
    label,
    loglikelihood,
    loglikelihood_gradient,
    update

include("base.jl")
include("features.jl")
include("util.jl")
end

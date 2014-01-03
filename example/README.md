# Example

This directory contains example code. The examples provided in this directory
are intendet to show the basic usage of the CRF package. They are not meant to
be an introduction to CRFs.
The examples center around artificial sequences of weather observations.
You can examine the data in the `weather.csv` file.
The `weather.csv` file contains 50 sequences of 50 labeled observations
each. The sequences were sampled from a HMM.

```csv
1.02044021484,2.22946587781,sunny
2.11016889919,0.995528238504,sunny
-1.27135072173,-0.458028479475,sunny
1.05372606377,-1.7191137792,sunny
3.39480499807,-5.67863590779,rainy
....
```

The following code loads the weather data

```julia
include("util.jl")

X, Y = load("weater.csv")
println(summary(X))
println(summary(Y))
```

and should print out

```
50-element Array{Array{Array{Float64,1},1},1}
50-element Array{Array{ASCIIString,1},1}
```

The following code snippets assume you already loaded the weather data.

## Feature Function

We'll start with defining the feature function. The feature function
represents a set of features. Features should be binary valued.
The feature function should return an array of features, i.e. binary values.
Though feature values should be binary, using a real numbered data type avoids
unnecessary conversions, e.g. returning {0.0, 1.0} instead of {0, 1} or {false, true}.

### Parameters

The following parameters are passed to the feature function:

* `yp` label of observation `x[t-1]`
* `yt` label of observation `x[t]`
* `x` array of all observations
* `t` position of current observation

The observation at `t=1` has no predecessor, therefore no `yp`. Most CRF papers
and implementations use a special `START` label as value of `yp` at `t=1`. However,
since the labels can be of any type, a special `START` label would either
require a type union or a user defined start label passed to the Sequence
constructor. Both solutions aren't really nice. Therefore we use a different
approach: two methods of the feature function:

* `method (yt, x, t)` called for `t=1`
* `method (yp, yt, x, t)` called for `t>1`

An implementation of a feature function might look like this:

```julia
typealias XT Array{Float64,1}
typealias YT ASCIIString

const labels = YT[ "sunny", "rainy", "foggy" ]

function weather_features(yt::YT, x::Array{XT,1}, t::Int32)
    weather_features("", yt, x, t)
end

function weather_features(yp::YT, yt::YT, x::Array{XT,1}, t::Int32)
    res = Array(Float64, 9)
    idx = 1
    for ypval in labels, ytval in labels
        res[idx] = ((yp == ypval) & (yt == ytval)) ; idx += 1
    end
    return res
end
```
For the moment we'll ignore the fact that this feature function doesn't use
our observations at all. This example is just meant to show you the basic
principles.

```julia
function weather_features(yp::YT, yt::YT, x::Array{XT,1}, t::Int32)
    res = Features(9)
    for ypval in labels, ytval in labels
        @append! res ((yp == ypval) & (yt == ytval))
    end
    return convert(Array{Float64,1}, res)
end
```

The file `features.jl` contains the feature function used in these examples.

## Parameter Estimation

After you defined your feature function, you want to use labeled training
data for parameter estimation. The training data consists of one or more
sequences of observations with corresponding sequences of desired labels.

Parameter estimation is done by maximizing the loglikelihood function. The
CRF package doesn't provide a function optimization algorithm. The following
example uses the Optim package for this purpose.
Since we are looking for parameters maximizing the loglikelihood function
and Optim.optimize performs function minimization, we negate the
loglikelihood and it's derivative.

Let us consider the case of a single observation and label sequence.
Parameter estimation is done by

```julia
x = X[1]
y = Y[1]

crf = Sequence(x, y, features)

function f(x::Vector)
    -loglikelihood(crf, Θ=x)
end

function g!(x::Vector, storage::Vector)
    storage[:] = -loglikelihood_gradient(crf, Θ=x)
end

opt = optimize(f, g!, crf, method = :l_bfgs)
```

Parameter estimation over multiple sequences works quite similar, since
`loglikelihood` and `loglikelihood_gradient` functions accept arrays of
sequences, too.

```julia
crfs = Sequence[ Sequence(x, y, features) for (x, y) in zip(X[1:5], Y[1:5]) ]

function f(x::Vector)
    -loglikelihood(crfs, Θ=x)
end

function g!(x::Vector, storage::Vector)
    storage[:] = -loglikelihood_gradient(crfs, Θ=x)
end

opt = optimize(f, g!, crfs, method = :l_bfgs)
```

The file `parameter_estimation.jl` contains a working example.

## Sequence Labeling

Now we show how to use the estimated parameters for labeling
unlabeled observations. A single sequence of observations can be labeled
using the following code:

```julia
const labels = [ "sunny", "rainy", "foggy" ] # label alphabet
const params = [ ... ] # estimated parameters

crf = Sequence(x, features, Θ=params, labels=labels)
y = label(crf)

# Print observation - label pairs
for (xi, yi) in zip(x, y)
    println(xi, yi)
end
```

Since the labels aren't observed, we omit passing `y` to the constructor.
We pass a set from which the labels are drawn as keyword `labels` instead.
The `label` function returns an array of labels.
Again, labeling multiple sequences is pretty similar to labeling a
single sequence.

```julia
crfs = Sequence[ Sequence(x, features, Θ=params, labels=labels) for x in X ]
Y = label(crfs)
```

Instead of returning a single array of labels, the `label` function returns
an array of label arrays for multiple sequences. Have a look at `labeling.jl`
to see a working example.

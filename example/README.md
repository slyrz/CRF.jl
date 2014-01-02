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

## Feature Function

We'll start with defining the *feature function*. The feature function
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
crf = Sequence(x, y, features)

function f(x::Vector)
    -loglikelihood(crf, Θ=x)
end

function g!(x::Vector, storage::Vector)
    storage[:] = -loglikelihood_gradient(crf, Θ=x)
end

opt = optimize(f, g!, crf, method = :l_bfgs)
```

## Sequence Labeling


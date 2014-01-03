# CRF.jl

The CRF package implements linear-chain Conditional Random Fields.
CRFs are a probabilistic framework for labeling sequential data.

## Quickstart

```julia
julia> using CRF
julia> crf = Sequence(x, y, features)
julia> loglikelihood(crf)
julia> loglikelihood_gradient(crf)
julia> label(crf)
```

The [example directory](example/) contains a detailed documentation.

## Further Reading

 - Charles Sutton, Andrew McCallum. *An Introduction to Conditional Random Fields for Relational Learning.* Introduction to Statistical Relational Learning, Vol. 93, pp. 142-146, 2007.

 - John Lafferty, Andrew McCallum, Fernando Pereira. *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* In Proceedings of the Eighteenth International Conference on Machine Learning (ICML-2001), 2001.

 - Hanna M. Wallach. *Conditional Random Fields: An Introduction.* Technical Report MS-CIS-04-21. Department of Computer and Information Science, University of Pennsylvania, 2004.

 - Thomas G. Dietterich. *Machine Learning for Sequential Data: A Review.* In Structural, Syntactic, and Statistical Pattern Recognition; Lecture Notes in Computer Science, Vol. 2396, T. Caelli (Ed.), pp. 15–30, Springer-Verlag, 2002.

More material on CRFs can be found [here](http://www.inference.phy.cam.ac.uk/hmw26/crf/).

using Base
using Base.Test
using CRF

import CRF.logsumexp, CRF.product, Base.rand

# Helper function
rand(lo::Int,hi::Int) = lo + abs(rand(Int) % hi)

# Test logsumexp function
@test logsumexp([0.0:9.0]) == logsumexp([0:9]) == 9.4586297444267107

# Test product iterator
l = (None,None,None)
for (x,y,z) in product(1:9, ['a', 'b', 'c'], 1:9)
    @test x in 1:9
    @test y in "abc"
    @test z in 1:9
    @test l != (x,y,z) # Something must have change

    l = (x,y,z)
end

N = 500
for i = 1:N
    itr = Range1{Int}[ rand(1,2):rand(1,7) for j = 1:rand(1,7) ]
    if any([ l == 0 for l in map(length,itr) ])
        @test_throws(p = product(tuple(itr...)))
    else
        m = prod(map(length, itr))
        n = 0
        for x in product(tuple(itr...))
            n += 1
        end
        @test n == m
    end
end

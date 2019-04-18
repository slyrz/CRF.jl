using Base
using Test
using CRF

v = Features(100)
w = Features(100)

# Check if adding features works like it should (you sould be able to access
# the value of global variables in the append! macro)
g = true
for i = 1:10, j = 1:10
    @append! v ((g) & ((i < 7) | (j > 4)))
end

g = false
for i = 1:10, j = 1:10
    @append! w ((g) & ((i < 7) | (j > 4)))
end

@test (v.i == 101) & (sum(v.x) == 84)
@test (w.i == 101) & (sum(w.x) == 0)

empty!(v)
empty!(w)
@test (v.i == 1) & (sum(v.x) == 0)
@test (w.i == 1) & (sum(w.x) == 0)

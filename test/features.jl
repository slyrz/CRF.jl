using Base
using Base.Test
using CRF

v = Features(100)
w = Features(100)

# Check if addind features works like it should (you sould be able to access
# the value of global variables in the anonymous functions)
g = true
append!(v, 1:10, 1:10) do x
    (g) & ((x[1] < 7) | (x[2] > 4))
end

g = false
append!(w, 1:10, 1:10) do x
    (g) & ((x[1] < 7) | (x[2] > 4))
end

@test (v.i == 101) & (sum(v.x) == 84)
@test (w.i == 101) & (sum(w.x) == 0)

empty!(v)
empty!(w)
@test (v.i == 1) & (sum(v.x) == 0)
@test (w.i == 1) & (sum(w.x) == 0)

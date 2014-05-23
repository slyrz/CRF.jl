using CRF

tests = [
    "base.jl",
    "features.jl",
]

for test in tests
    println(">>> $(test)")
    include(test)
end

using CRF

include("util.jl")
include("features.jl")

const output_succ = colorize(:green, ".")
const output_fail = colorize(:red, "!")

# These parameters were estimated using sequences 1:5 of the weather data
const weights = [
    -0.28026, -0.54288,  0.04901, -0.25603,  1.09337,  0.49870,
     0.13389, -0.23739, -0.24375, -1.08821, -0.06286,  0.89308,
    -0.37372,  0.38654,  0.38167, -3.53123,  1.76168,  3.07147,
    -1.50942,  0.51421, -4.12856, -0.64580,  4.74698, -0.31943,
     2.64340, -0.58929, -2.62894,  0.10829,  0.18280, -0.54266,
    -0.47681,  2.13713, -1.66658, -2.48922,  1.82307, -0.30672,
     1.82972,  0.71780, -2.07108, -0.51137,  0.60169, -0.13208,
    -1.85513,  0.43814, -1.48759,  0.66490,  2.30342,  0.42272,
     1.62713,  0.70396, -1.13422,  3.62780,  4.14362,  1.62987,
    -0.85902, -3.39346, -3.32714, -2.12237, -1.70612, -0.31701,
     0.28682, -0.53939, -3.27933, -4.02713, -2.32070,  1.53830,
     2.50143,  1.38448,  1.65289, -0.06920,  0.63677,  0.05326,
     1.14765, -0.65981, -0.98722, -0.32634, -0.84214,  0.73448,
     1.06528,  0.95661,  0.84304, -0.59334, -0.36241, -0.66310,
     1.45682,  6.45608,  1.91501,  4.23405,  3.87607, -0.46321,
    -1.01568, -2.34954, -1.60510, -0.80973, -2.09774, -0.23512,
    -0.96984,  0.29281, -1.69349, -1.77548, -0.18955, -1.13785,
     0.13768, -2.34054,  0.25428, -1.02766, -0.30154, -0.35736,
    -0.48482,  0.05892, -0.01486, -0.04475,  1.87549, -4.50467,
    -1.51220, -0.85258, -1.91109,  3.69303,  1.07203,  4.04240,
     1.01121,  2.16412, -0.29784,  0.23954, -0.66587,  0.89715,
    -0.31191, -0.08533, -0.15711, -1.60186, -0.02909, -0.69339,
    -0.40224, -0.50869,  0.00055, -0.99869,  0.05103, -0.60559,
     0.44725,  0.13268, -2.81193, -1.61552, -1.24948, -2.67640,
    -1.28247, -2.80358, -0.34114, -0.74070,  1.51590, -1.77853,
     1.70655, -0.54594,  1.30247, -0.46603,  2.28510,  1.73859,
    -0.34424,  2.82764, -0.23264,  3.28881, -0.02248,  2.36526,
     0.10379,  1.30489,  0.37751,  0.86419,  0.42223,  0.25484,
]


# Load weather data
X, Y = load("weather.csv")

# Remove sequences we used for parameter estimation
splice!(X, 1:5)
splice!(Y, 1:5)

# Create sequences
crfs = Sequence[ Sequence(x, features, Θ=weights, labels=labels) for x in X ]

# Calling label on the array of sequences does the trick
Y_true = Y
Y_pred = label(crfs)

# Let's compare the predicted labels with the hidden labels
error = 0
total = 0
for (y_true, y_pred) in zip(Y_true, Y_pred)
    for (lt, lp) in zip(y_true, y_pred)
        print((lt == lp) ? output_succ : output_fail)
        total += 1
        error += (lt != lp)
    end
    print(" ")
end
print("\n")

@printf "Total: %6d\n" total
@printf "Error: %6d (%.2f %%)\n" error ((error / total) * 100.0)

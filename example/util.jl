# Function to load data from CSV file. Empty lines indicate end of a sequence.
function load(path::ASCIIString)
    stream = open(path, "r")
    X = Array{XT,1}[ XT[ ] ]
    Y = Array{YT,1}[ YT[ ] ]
    for line in map(strip, eachline(stream))
        if isempty(line)
            push!(X, XT[ ])
            push!(Y, YT[ ])
        else
            x1, x2, y = split(line, ',')
            push!(X[end], [ float(x1), float(x2) ])
            push!(Y[end], y)
        end
    end
    close(stream)
    return X, Y
end

function colorize(col::Symbol, str::ASCIIString)
    get(Base.text_colors, col, :normal) * str * get(Base.text_colors, :normal, :normal)
end

module JaynesScratch

include("../src/Gen.jl")
using .Gen
using Distributions

model = @jaynes () -> begin
    x ~ Normal(0.0, 1.0)
    x
end

tr = simulate(model, ())
display(tr)
tg = target([(:x, ) => 1.0])
tr, w, rd, d = update(tr, (), (), tg)
display(tr)

end # module

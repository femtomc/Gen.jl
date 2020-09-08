module JaynesScratch

include("../src/Gen.jl")
using .Gen
using Distributions

model = @jaynes () -> begin
    x ~ Normal(0.0, 1.0)
    y ~ Normal(x, 5.0)
    y
end

tr = simulate(model, ())
display(tr)
tg = target([(:y, ) => 1.0])
trs, lnw, lmle = importance_sampling(model, (), tg, 5000)
map(trs) do tr
    display(tr)
end

end # module

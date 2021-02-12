# Incremental DSL

### Design 1

A lightweight DSL around the existing incremental computing extension mechanisms. This DSL would mostly serve to eliminate boilerplate.
E.g. from example in docs:

```julia
struct MyState
    prev_arr::Vector{Float64}
    sum::Float64
end

struct MySum <: CustomUpdateGF{Float64,MyState} end

function Gen.apply_with_state(::MySum, args)
    arr = args[1]
    s = sum(arr)
    state = MyState(arr, s)
    (s, state)
end

function Gen.update_with_state(::MySum, state, args, argdiffs::Tuple{VectorDiff})
    arr = args[1]
    prev_sum = state.sum
    retval = prev_sum
    for i in keys(argdiffs[1].updated)
        retval += (arr[i] - state.prev_arr[i])
    end
    prev_length = length(state.prev_arr)
    new_length = length(arr)
    for i=prev_length+1:new_length
        retval += arr[i]
    end
    for i=new_length+1:prev_length
        retval -= arr[i]
    end
    state = MyState(arr, retval)
    (state, retval, UnknownChange())
end

Gen.num_args(::MySum) = 1
```

Here, we create a new `CustomUpdateGF{Float64, MyState}` to deal with updating a vector sum. So the workflow is:

`(incremental function) -> required state -> GF`

We could remove some boilerplate by defining a macro `@incremental` which accepts a state struct:

```julia
@incremental begin
    target::Vector{Float64}
    @tracked (sum::Float64) sum
    @tracked (foldr_prod::Float64) arr -> foldr(*, arr)
end
```

Here, the user uses `@incremental` as a "summary" of the different interface functions which operate on the type of `target` (here, a `Vector{Float64}`). This macro would expand and eliminate boilerplate in `apply_with_state`.

```julia
struct MyState_sum
    target::Vector{Float64}
    sum::Float64
end

struct MyState_foldr_prod
    target::Vector{Float64}
    foldr_prod::Float64
end

struct GF_MyState_sum <: CustomUpdateGF{Float64,MyState_sum} end
Gen.num_args(::GF_MyState_sum) = 1

struct GF_MyState_foldr_prod <: CustomUpdateGF{Float64,MyState_foldr_prod} end
Gen.num_args(::GF_MyState_foldr_prod) = 1

function Gen.apply_with_state(::MyState_sum, args)
    arr = args[1]
    s = sum(arr)
    state = MyState_sum(arr, s)
    (s, state)
end

function Gen.apply_with_state(::MyState_foldr_prod, args)
    arr = args[1]
    s = (arr -> foldr(*, arr))(arr)
    state = MyState_sum(arr, s)
    (s, state)
end
```

### Design 2

We design a miniature DSL which generates generative functions for all primitive accessors/mutators for a struct. E.g.:

```julia
@incremental struct MyFoo
    x::Float64
    y::Vector{Float64}
end
```

Here, this macro expands to generate a constructor, and a set of accessors:

```julia
struct MyFoo
    x::Float64
    y::Vector{Float64}
end

struct MyFooDiff
    x_diff :: ScalarDiff
    y_diff :: VectorDiff
end

struct MyFoo_constructor <: CustomConstructorGF{MyFoo} end

struct MyFoo_get_x <: CustomAccessorGF{Float64, MyFoo} end
# TODO: this is not correct yet.
function traceat(::UpdateState, MyFoo_get_x)
    return getproperty(get_state(MyFoo_get_x), :x), NoChange()
end

struct MyFoo_get_y <: CustomAccessorGF{Vector{Float64}, MyFoo} end
# TODO: this is not correct yet.
function traceat(::UpdateState, MyFoo_get_y)
    return getproperty(get_state(MyFoo_get_y), :y), NoChange()
end
```

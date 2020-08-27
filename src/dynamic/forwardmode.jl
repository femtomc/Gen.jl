mutable struct GFForwardModeState{T <: Tuple,
                                  D}
    target::T
    tr::Trace
    weight::D
    visited::AddressVisitor
    params::Dict{Symbol, Any}
end

function GFForwardModeState(key, tr, weight)
    ForwardModeContext(key, 
                       tr, 
                       weight, 
                       Visitor(), 
                       Dict{Symbol, Any}())
end

function GFForwardModeState(key, params, tr, weight)
    ForwardModeContext(key, 
                       tr, 
                       weight, 
                       AddressVisitor(), 
                       params)
end

@inline state_getindex(state, tr, key) = (key, ) == state.target ? Dual(getindex(tr, key), 1.0) : getindex(tr, key)

function traceat(state::GFForwardModeState, 
                 dist::Distribution{T},
                 args, 
                 key)

    # check that key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key - if key == target, this is a dual number.
    retval = state_getindex(state, tr, key)

    # compute logpdf
    score = logpdf(dist, retval, args...)

    # increment weight
    state.weight += score

    retval
end

function traceat(state::GFForwardModeState, 
                 gen_fn::GenerativeFunction{T,U},
                 args, 
                 key) where {T,U}

    local subtrace::U

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constraints = get_submap(state.constraints, key)

    # get subtrace and weight, weight will be a dual if key matches target.
    (subtrace, weight) = generate(gen_fn, args, constraints)

    # update weight
    state.weight += weight

    # get return value
    retval = get_retval(subtrace)
    retval
end

function splice(state::GFForwardModeState, 
                gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function forward(target, params, tr, seed)
    state = GFForwardMode(target, params, cl, seed)
    retval = exec(gen_fn, state, args)
    (state.trace, state.weight)
end

function forward(target, tr, seed)
    state = GFForwardMode(target, params, tr, seed)
    retval = exec(gen_fn, state, args)
    (state.trace, state.weight)
end

mutable struct GFForwardModeState{T <: Tuple,
                                  D}
    target::T
    map::ChoiceMap
    weight::D
    visited::AddressVisitor
    params::Dict{Symbol, Any}
end

function GFForwardModeState(addr, tr, weight)
    ForwardModeContext(addr, tr, weight, Visitor(), Empty())
end

function GFForwardModeState(addr, params, tr, weight)
    ForwardModeContext(addr, tr, weight, AddressVisitor(), params)
end

function traceat(state::GFGenerateState, gen_fn::GenerativeFunction{T,U},
                 args, key) where {T,U}

    local subtrace::U
    local retval::T

    # check key was not already visited, and mark it as visited
    visit!(state.visitor, key)

    # check for constraints at this key
    constraints = get_submap(state.constraints, key)

    # get subtrace and weight, weight will be a dual if addr matches target.
    (subtrace, weight) = generate(gen_fn, args, constraints)

    # update weight
    state.weight += weight

    # get return value
    retval = get_retval(subtrace)
    retval
end

function splice(state::GFGenerateState, gen_fn::DynamicDSLFunction,
                args::Tuple)
    prev_params = state.params
    state.params = gen_fn.params
    retval = exec(gen_fn, state, args)
    state.params = prev_params
    retval
end

function forward(addr, params, cl, seed)
    ctx = ForwardMode(addr, params, cl, seed)
    ret = ctx(cl.fn, cl.args...)
    ret, ctx.weight
end

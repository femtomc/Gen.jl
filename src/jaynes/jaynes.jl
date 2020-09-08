using Jaynes

# ------------ DSL implementation for Jaynes ------------ #

# Utilities.
is_choice(::Jaynes.Choice) = true
is_choice(a) = false
is_value(::Jaynes.Value) = true
is_value(a) = false

# ------------ Selections ------------ #

struct JaynesSelection{K <: Jaynes.AddressMap{Jaynes.Select}} <: Selection
    sel::K
end
unwrap(js::JaynesSelection) = js.sel

Base.in(addr, selection::JaynesSelection) = haskey(selection.sel, addr)
Base.getindex(selection::JaynesSelection, addr) = Jaynes.get_sub(selection.sel, addr)
Base.isempty(selection::JaynesSelection, addr) = isempty(selection.sel, addr)

# ------------ Choice map ------------ #

struct JaynesChoiceMap{K <: Jaynes.AddressMap} <: ChoiceMap
    chm::K
end
unwrap(jcm::JaynesChoiceMap) = jcm.chm

has_value(choices::JaynesChoiceMap, addr) = Jaynes.has_value(choices.chm, addr)
get_value(choices::JaynesChoiceMap, addr) = Jaynes.get_value(choices.chm, addr)
get_submap(choices::JaynesChoiceMap, addr) = Jaynes.get_sub(choices.chm, addr)
get_values_shallow(choices::JaynesChoiceMap, addr) = Jaynes.shallow_iterator(choices.chm)
to_array(choices::JaynesChoiceMap, ::Type{T}) where T = Jaynes.array(choices.chm, T)
from_array(choices::JaynesChoiceMap, arr::Vector) = Jaynes.target(choices.chm, arr)

target(c::Vector{Pair{T, K}}) where {T <: Tuple, K} = JaynesChoiceMap(Jaynes.target(c))

# Trace.
mutable struct JaynesTrace{T, K <: Jaynes.CallSite} <: Trace
    gen_fn::T
    record::K
    isempty::Bool
end

set_retval!(trace::JaynesTrace, retval) = (trace.retval = retval)

has_choice(trace::JaynesTrace, addr) = haskey(trace.record, addr) && is_choice(get_sub(trace.record, addr))

function get_choice(trace::JaynesTrace, addr)
    ch = Jaynes.get_sub(trace.record, addr)
    !is_choice(ch) && throw(KeyError(addr))
    ch
end

Base.display(jtr::JaynesTrace) = Base.display(Jaynes.get_trace(jtr.record))

# Trace GFI methods.
get_args(trace::JaynesTrace) = Jaynes.get_args(trace.record)
get_retval(trace::JaynesTrace) = Jaynes.get_retval(trace.record)
get_score(trace::JaynesTrace) = Jaynes.get_score(trace.record)
get_gen_fn(trace::JaynesTrace) = trace.gen_fn

# Generative function.
struct JaynesFunction{N, R} <: GenerativeFunction{R, JaynesTrace}
    fn::Function
    arg_types::NTuple{N, Type}
    has_argument_grads::NTuple{N, Bool}
    accepts_output_grad::Bool
end

function JaynesFunction(arg_types::NTuple{N, Type},
                        func::Function,
                        has_argument_grads::NTuple{N, Bool},
                        accepts_output_grad::Bool,
                        ::Type{R}) where {N, R}
    JaynesFunction{N, R}(func, arg_types, has_argument_grads, accepts_output_grad)
end

function (jfn::JaynesFunction)(args...)
    jfn.fn(args...)
end

has_argument_grads(jfn::JaynesFunction) = jfn.has_argument_grads

# ------------ Model GFI interface ------------ #

function simulate(jfn::JaynesFunction, args::Tuple)
    ret, cl = Jaynes.simulate(jfn.fn, args...)
    JaynesTrace(jfn, cl, false)
end

function generate(jfn::JaynesFunction, args::Tuple, chm::JaynesChoiceMap)
    ret, cl, w = Jaynes.generate(unwrap(chm), jfn.fn, args...)
    JaynesTrace(jfn, cl, false), w
end

function assess(jfn::JaynesFunction, args::Tuple, choices::JaynesChoiceMap)
    ret, w = Jaynes.score(unwrap(choices), jfn.fn, args...)
    w, ret
end

function propose(jfn::JaynesFunction, args::Tuple)
    ret, cl, w = Jaynes.propose(jfn.fn, args...)
    Jaynes.get_trace(cl), w, ret
end

function update(trace::JaynesTrace, args::Tuple, arg_diffs::Tuple, constraints::JaynesChoiceMap)
    ret, cl, w, d, rd = Jaynes.update(unwrap(constraints), trace.record, args, arg_diffs)
    JaynesTrace(get_gen_fn(trace), cl, false), w, UnknownChange(), d
end

function regenerate(trace::JaynesTrace, args::Tuple, arg_diffs::Tuple, selection::JaynesSelection)
    ret, cl, w, d, rd = Jaynes.regenerate(unwrap(selection), trace.record, args, arg_diffs)
    JaynesTrace(get_gen_fn(trace), cl, false), w, UnknownChange(), d
end

# ------------ Convenience macro ------------ #

macro jaynes(expr)
    def = Jaynes._sugar(expr)
    if @capture(def, function decl_(args__) body__ end)
        trans = quote 
            $def
            JaynesFunction((), $decl, (), true, Any)
        end
    else
        trans = quote
            JaynesFunction((), $def, (), true, Any)
        end
    end
    esc(trans)
end

# ------------ exports ------------ #

export JaynesFunction
export JaynesTrace
export target

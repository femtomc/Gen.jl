using Jaynes

# ------------ DSL implementation for Jaynes ------------ #

# Utilities.
is_choice(::Jaynes.Choice) = true
is_choice(a) = false
is_value(::Jaynes.Value) = true
is_value(a) = false

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

function generate(jfn::JaynesFunction, args::Tuple)
    ret, cl, w = Jaynes.generate(jfn.fn, args...)
    JaynesTrace(jfn, cl, false), w
end

function assess(jfn::JaynesFunction, args::Tuple, choices::C) where C <: Jaynes.AddressMap
    ret, w = Jaynes.score(choices, jfn.fn, args...)
    w, ret
end

function propose(jfn::JaynesFunction, args::Tuple)
    ret, cl, w = Jaynes.propose(jfn.fn, args...)
    Jaynes.get_trace(cl), w, ret
end

function update(trace::JaynesTrace, args::Tuple, arg_diffs::Tuple, constraints::C) where C <: Jaynes.AddressMap
    ret, cl, w, d, rd = Jaynes.update(constraints, trace.record, args, arg_diffs)
    JaynesTrace(get_gen_fn(trace), cl, false), w, UnknownChange(), d
end

function regenerate(trace::JaynesTrace, args::Tuple, arg_diffs::Tuple, selection::C) where C <: Jaynes.Target
    ret, cl, w, d, rd = Jaynes.regenerate(selection, trace.record, args, arg_diffs)
    JaynesTrace(get_gen_fn(trace), cl, false), w, UnknownChange(), d
end

# ------------ Selections ------------ #

struct JaynesSelection{K <: Jaynes.AddressMap{Jaynes.Select}} <: Selection
    sel::K
end

Base.in(addr, selection::JaynesSelection) = haskey(selection.sel, addr)
Base.getindex(selection::JaynesSelection, addr) = Jaynes.get_sub(selection.sel, addr)
Base.isempty(selection::JaynesSelection, addr) = isempty(selection.sel, addr)

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

import Jaynes.target
export target

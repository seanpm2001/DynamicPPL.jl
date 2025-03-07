using Distributions:
    UnivariateDistribution, MultivariateDistribution, MatrixDistribution, Distribution

const AMBIGUITY_MSG =
    "Ambiguous `LHS .~ RHS` or `@. LHS ~ RHS` syntax. The broadcasting " *
    "can either be column-wise following the convention of Distributions.jl or " *
    "element-wise following Julia's general broadcasting semantics. Please make sure " *
    "that the element type of `LHS` is not a supertype of the support type of " *
    "`AbstractVector` to eliminate ambiguity."

alg_str(spl::Sampler) = string(nameof(typeof(spl.alg)))

# utility funcs for querying sampler information
require_gradient(spl::Sampler) = false
require_particles(spl::Sampler) = false

# Allows samplers, etc. to hook into the final logp accumulation in the tilde-pipeline.
function acclogp_assume!!(context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_assume!!(NodeTrait(acclogp_assume!!, context), context, vi, logp)
end
function acclogp_assume!!(::IsParent, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_assume!!(childcontext(context), vi, logp)
end
function acclogp_assume!!(::IsLeaf, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp!!(context, vi, logp)
end

function acclogp_observe!!(context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_observe!!(NodeTrait(acclogp_observe!!, context), context, vi, logp)
end
function acclogp_observe!!(::IsParent, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp_observe!!(childcontext(context), vi, logp)
end
function acclogp_observe!!(::IsLeaf, context::AbstractContext, vi::AbstractVarInfo, logp)
    return acclogp!!(context, vi, logp)
end

# assume
"""
    tilde_assume(context::SamplingContext, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value with a context associated
with a sampler.

Falls back to
```julia
tilde_assume(context.rng, context.context, context.sampler, right, vn, vi)
```
"""
function tilde_assume(context::SamplingContext, right, vn, vi)
    return tilde_assume(context.rng, context.context, context.sampler, right, vn, vi)
end

# Leaf contexts
function tilde_assume(context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), context, args...)
end
function tilde_assume(::IsLeaf, context::AbstractContext, right, vn, vi)
    return assume(right, vn, vi)
end
function tilde_assume(::IsParent, context::AbstractContext, args...)
    return tilde_assume(childcontext(context), args...)
end

function tilde_assume(rng::Random.AbstractRNG, context::AbstractContext, args...)
    return tilde_assume(NodeTrait(tilde_assume, context), rng, context, args...)
end
function tilde_assume(
    ::IsLeaf, rng::Random.AbstractRNG, context::AbstractContext, sampler, right, vn, vi
)
    return assume(rng, sampler, right, vn, vi)
end
function tilde_assume(
    ::IsParent, rng::Random.AbstractRNG, context::AbstractContext, args...
)
    return tilde_assume(rng, childcontext(context), args...)
end

function tilde_assume(::LikelihoodContext, right, vn, vi)
    return assume(nodist(right), vn, vi)
end
function tilde_assume(rng::Random.AbstractRNG, ::LikelihoodContext, sampler, right, vn, vi)
    return assume(rng, sampler, nodist(right), vn, vi)
end

function tilde_assume(context::PrefixContext, right, vn, vi)
    return tilde_assume(context.context, right, prefix(context, vn), vi)
end
function tilde_assume(
    rng::Random.AbstractRNG, context::PrefixContext, sampler, right, vn, vi
)
    return tilde_assume(rng, context.context, sampler, right, prefix(context, vn), vi)
end

"""
    tilde_assume!!(context, right, vn, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value and updated `vi`.

By default, calls `tilde_assume(context, right, vn, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_assume!!(context, right, vn, vi)
    value, logp, vi = tilde_assume(context, right, vn, vi)
    return value, acclogp_assume!!(context, vi, logp)
end

# observe
"""
    tilde_observe(context::SamplingContext, right, left, vi)

Handle observed constants with a `context` associated with a sampler.

Falls back to `tilde_observe(context.context, context.sampler, right, left, vi)`.
"""
function tilde_observe(context::SamplingContext, right, left, vi)
    return tilde_observe(context.context, context.sampler, right, left, vi)
end

# Leaf contexts
function tilde_observe(context::AbstractContext, args...)
    return tilde_observe(NodeTrait(tilde_observe, context), context, args...)
end
tilde_observe(::IsLeaf, context::AbstractContext, args...) = observe(args...)
function tilde_observe(::IsParent, context::AbstractContext, args...)
    return tilde_observe(childcontext(context), args...)
end

tilde_observe(::PriorContext, right, left, vi) = 0, vi
tilde_observe(::PriorContext, sampler, right, left, vi) = 0, vi

# `MiniBatchContext`
function tilde_observe(context::MiniBatchContext, right, left, vi)
    logp, vi = tilde_observe(context.context, right, left, vi)
    return context.loglike_scalar * logp, vi
end
function tilde_observe(context::MiniBatchContext, sampler, right, left, vi)
    logp, vi = tilde_observe(context.context, sampler, right, left, vi)
    return context.loglike_scalar * logp, vi
end

# `PrefixContext`
function tilde_observe(context::PrefixContext, right, left, vi)
    return tilde_observe(context.context, right, left, vi)
end
function tilde_observe(context::PrefixContext, sampler, right, left, vi)
    return tilde_observe(context.context, sampler, right, left, vi)
end

"""
    tilde_observe!!(context, right, left, vname, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value and updated `vi`.

Falls back to `tilde_observe!!(context, right, left, vi)` ignoring the information about variable name
and indices; if needed, these can be accessed through this function, though.
"""
function tilde_observe!!(context, right, left, vname, vi)
    return tilde_observe!!(context, right, left, vi)
end

"""
    tilde_observe(context, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.

By default, calls `tilde_observe(context, right, left, vi)` and accumulates the log
probability of `vi` with the returned value.
"""
function tilde_observe!!(context, right, left, vi)
    logp, vi = tilde_observe(context, right, left, vi)
    return left, acclogp_observe!!(context, vi, logp)
end

function assume(rng, spl::Sampler, dist)
    return error("DynamicPPL.assume: unmanaged inference algorithm: $(typeof(spl))")
end

function observe(spl::Sampler, weight)
    return error("DynamicPPL.observe: unmanaged inference algorithm: $(typeof(spl))")
end

# fallback without sampler
function assume(dist::Distribution, vn::VarName, vi)
    r, logp = invlink_with_logpdf(vi, vn, dist)
    return r, logp, vi
end

# TODO: Remove this thing.
# SampleFromPrior and SampleFromUniform
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    vi::VarInfoOrThreadSafeVarInfo,
)
    if haskey(vi, vn)
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if sampler isa SampleFromUniform || is_flagged(vi, vn, "del")
            # TODO(mhauru) Is it important to unset the flag here? The `true` allows us
            # to ignore the fact that for VarNamedVector this does nothing, but I'm unsure
            # if that's okay.
            unset_flag!(vi, vn, "del", true)
            r = init(rng, dist, sampler)
            f = to_maybe_linked_internal_transform(vi, vn, dist)
            # TODO(mhauru) This should probably be call a function called setindex_internal!
            # Also, if we use !! we shouldn't ignore the return value.
            BangBang.setindex!!(vi, f(r), vn)
            setorder!(vi, vn, get_num_produce(vi))
        else
            # Otherwise we just extract it.
            r = vi[vn, dist]
        end
    else
        r = init(rng, dist, sampler)
        if istrans(vi)
            f = to_linked_internal_transform(vi, dist)
            push!!(vi, vn, f(r), dist, sampler)
            # By default `push!!` sets the transformed flag to `false`.
            settrans!!(vi, true, vn)
        else
            push!!(vi, vn, r, dist, sampler)
        end
    end

    # HACK: The above code might involve an `invlink` somewhere, etc. so we need to correct.
    logjac = logabsdetjac(istrans(vi, vn) ? link_transform(dist) : identity, r)
    return r, logpdf(dist, r) - logjac, vi
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
observe(sampler::AbstractSampler, right, left, vi) = observe(right, left, vi)
function observe(right::Distribution, left, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(right, left), vi
end

# .~ functions

# assume
"""
    dot_tilde_assume(context::SamplingContext, right, left, vn, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value for a context
associated with a sampler.

Falls back to
```julia
dot_tilde_assume(context.rng, context.context, context.sampler, right, left, vn, vi)
```
"""
function dot_tilde_assume(context::SamplingContext, right, left, vn, vi)
    return dot_tilde_assume(
        context.rng, context.context, context.sampler, right, left, vn, vi
    )
end

# `DefaultContext`
function dot_tilde_assume(context::AbstractContext, args...)
    return dot_tilde_assume(NodeTrait(dot_tilde_assume, context), context, args...)
end
function dot_tilde_assume(rng::Random.AbstractRNG, context::AbstractContext, args...)
    return dot_tilde_assume(NodeTrait(dot_tilde_assume, context), rng, context, args...)
end

function dot_tilde_assume(::IsLeaf, ::AbstractContext, right, left, vns, vi)
    return dot_assume(right, left, vns, vi)
end
function dot_tilde_assume(::IsLeaf, rng, ::AbstractContext, sampler, right, left, vns, vi)
    return dot_assume(rng, sampler, right, vns, left, vi)
end

function dot_tilde_assume(::IsParent, context::AbstractContext, args...)
    return dot_tilde_assume(childcontext(context), args...)
end
function dot_tilde_assume(::IsParent, rng, context::AbstractContext, args...)
    return dot_tilde_assume(rng, childcontext(context), args...)
end

function dot_tilde_assume(
    rng::Random.AbstractRNG, ::DefaultContext, sampler, right, left, vns, vi
)
    return dot_assume(rng, sampler, right, vns, left, vi)
end

# `LikelihoodContext`
function dot_tilde_assume(context::LikelihoodContext, right, left, vn, vi)
    return dot_assume(nodist(right), left, vn, vi)
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, context::LikelihoodContext, sampler, right, left, vn, vi
)
    return dot_assume(rng, sampler, nodist(right), vn, left, vi)
end

# `PrefixContext`
function dot_tilde_assume(context::PrefixContext, right, left, vn, vi)
    return dot_tilde_assume(context.context, right, left, prefix.(Ref(context), vn), vi)
end

function dot_tilde_assume(
    rng::Random.AbstractRNG, context::PrefixContext, sampler, right, left, vn, vi
)
    return dot_tilde_assume(
        rng, context.context, sampler, right, left, prefix.(Ref(context), vn), vi
    )
end

"""
    dot_tilde_assume!!(context, right, left, vn, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value and updated `vi`.

Falls back to `dot_tilde_assume(context, right, left, vn, vi)`.
"""
function dot_tilde_assume!!(context, right, left, vn, vi)
    value, logp, vi = dot_tilde_assume(context, right, left, vn, vi)
    return value, acclogp_assume!!(context, vi, logp), vi
end

# `dot_assume`
function dot_assume(
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    vi::AbstractVarInfo,
)
    @assert length(dist) == size(var, 1) "dimensionality of `var` ($(size(var, 1))) is incompatible with dimensionality of `dist` $(length(dist))"
    # NOTE: We cannot work with `var` here because we might have a model of the form
    #
    #     m = Vector{Float64}(undef, n)
    #     m .~ Normal()
    #
    # in which case `var` will have `undef` elements, even if `m` is present in `vi`.
    r = vi[vns, dist]
    lp = sum(zip(vns, eachcol(r))) do (vn, ri)
        return Bijectors.logpdf_with_trans(dist, ri, istrans(vi, vn))
    end
    return r, lp, vi
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::AbstractVarInfo,
)
    @assert length(dist) == size(var, 1)
    r = get_and_set_val!(rng, vi, vns, dist, spl)
    lp = sum(Bijectors.logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    return r, lp, vi
end

function dot_assume(
    dist::Distribution, var::AbstractArray, vns::AbstractArray{<:VarName}, vi
)
    r = getindex.((vi,), vns, (dist,))
    lp = sum(Bijectors.logpdf_with_trans.((dist,), r, istrans.((vi,), vns)))
    return r, lp, vi
end

function dot_assume(
    dists::AbstractArray{<:Distribution},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    vi,
)
    r = getindex.((vi,), vns, dists)
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans.((vi,), vns)))
    return r, lp, vi
end

function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::AbstractVarInfo,
)
    r = get_and_set_val!(rng, vi, vns, dists, spl)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans.((vi,), vns)))
    return r, lp, vi
end
function dot_assume(rng, spl::Sampler, ::Any, ::AbstractArray{<:VarName}, ::Any, ::Any)
    return error(
        "[DynamicPPL] $(alg_str(spl)) doesn't support vectorizing assume statement"
    )
end

# HACK: These methods are only used in the `get_and_set_val!` methods below.
# FIXME: Remove these.
function _link_broadcast_new(vi, vn, dist, r)
    b = to_linked_internal_transform(vi, dist)
    return b(r)
end

function _maybe_invlink_broadcast(vi, vn, dist)
    xvec = getindex_internal(vi, vn)
    b = from_maybe_linked_internal_transform(vi, vn, dist)
    return b(xvec)
end

function get_and_set_val!(
    rng,
    vi::VarInfoOrThreadSafeVarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    n = length(vns)
    if haskey(vi, vns[1])
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vns[1], "del")
            # TODO(mhauru) Is it important to unset the flag here? The `true` allows us
            # to ignore the fact that for VarNamedVector this does nothing, but I'm unsure if
            # that's okay.
            unset_flag!(vi, vns[1], "del", true)
            r = init(rng, dist, spl, n)
            for i in 1:n
                vn = vns[i]
                f_link_maybe = to_maybe_linked_internal_transform(vi, vn, dist)
                setindex!!(vi, f_link_maybe(r[:, i]), vn)
                setorder!(vi, vn, get_num_produce(vi))
            end
        else
            r = vi[vns, dist]
        end
    else
        r = init(rng, dist, spl, n)
        for i in 1:n
            vn = vns[i]
            if istrans(vi)
                ri_linked = _link_broadcast_new(vi, vn, dist, r[:, i])
                push!!(vi, vn, ri_linked, dist, spl)
                # `push!!` sets the trans-flag to `false` by default.
                settrans!!(vi, true, vn)
            else
                push!!(vi, vn, r[:, i], dist, spl)
            end
        end
    end
    return r
end

function get_and_set_val!(
    rng,
    vi::VarInfoOrThreadSafeVarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    if haskey(vi, vns[1])
        # Always overwrite the parameters with new ones for `SampleFromUniform`.
        if spl isa SampleFromUniform || is_flagged(vi, vns[1], "del")
            # TODO(mhauru) Is it important to unset the flag here? The `true` allows us
            # to ignore the fact that for VarNamedVector this does nothing, but I'm unsure if
            # that's okay.
            unset_flag!(vi, vns[1], "del", true)
            f = (vn, dist) -> init(rng, dist, spl)
            r = f.(vns, dists)
            for i in eachindex(vns)
                vn = vns[i]
                dist = dists isa AbstractArray ? dists[i] : dists
                f_link_maybe = to_maybe_linked_internal_transform(vi, vn, dist)
                setindex!!(vi, f_link_maybe(r[i]), vn)
                setorder!(vi, vn, get_num_produce(vi))
            end
        else
            rs = _maybe_invlink_broadcast.((vi,), vns, dists)
            r = reshape(rs, size(vns))
        end
    else
        f = (vn, dist) -> init(rng, dist, spl)
        r = f.(vns, dists)
        # TODO: This will inefficient since it will allocate an entire vector.
        # We could either:
        # 1. Figure out the broadcast size and use a `foreach`.
        # 2. Define an anonymous function which returns `nothing`, which
        #    we then broadcast. This will allocate a vector of `nothing` though.
        if istrans(vi)
            push!!.((vi,), vns, _link_broadcast_new.((vi,), vns, dists, r), dists, (spl,))
            # NOTE: Need to add the correction.
            # FIXME: This is not great.
            acclogp_assume!!(vi, sum(logabsdetjac.(link_transform.(dists), r)))
            # `push!!` sets the trans-flag to `false` by default.
            settrans!!.((vi,), true, vns)
        else
            push!!.((vi,), vns, r, dists, (spl,))
        end
    end
    return r
end

function set_val!(
    vi::VarInfoOrThreadSafeVarInfo,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    val::AbstractMatrix,
)
    @assert size(val, 2) == length(vns)
    foreach(enumerate(vns)) do (i, vn)
        setindex!!(vi, val[:, i], vn)
    end
    return val
end
function set_val!(
    vi::VarInfoOrThreadSafeVarInfo,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    val::AbstractArray,
)
    @assert size(val) == size(vns)
    foreach(CartesianIndices(val)) do ind
        setindex!!(vi, tovec(val[ind]), vns[ind])
    end
    return val
end

# observe
"""
    dot_tilde_observe(context::SamplingContext, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value for a context associated with a sampler.

Falls back to `dot_tilde_observe(context.context, context.sampler, right, left, vi)`.
"""
function dot_tilde_observe(context::SamplingContext, right, left, vi)
    return dot_tilde_observe(context.context, context.sampler, right, left, vi)
end

# Leaf contexts
function dot_tilde_observe(context::AbstractContext, args...)
    return dot_tilde_observe(NodeTrait(tilde_observe, context), context, args...)
end
dot_tilde_observe(::IsLeaf, ::AbstractContext, args...) = dot_observe(args...)
function dot_tilde_observe(::IsParent, context::AbstractContext, args...)
    return dot_tilde_observe(childcontext(context), args...)
end

dot_tilde_observe(::PriorContext, right, left, vi) = 0, vi
dot_tilde_observe(::PriorContext, sampler, right, left, vi) = 0, vi

# `MiniBatchContext`
function dot_tilde_observe(context::MiniBatchContext, right, left, vi)
    logp, vi = dot_tilde_observe(context.context, right, left, vi)
    return context.loglike_scalar * logp, vi
end

# `PrefixContext`
function dot_tilde_observe(context::PrefixContext, right, left, vi)
    return dot_tilde_observe(context.context, right, left, vi)
end

"""
    dot_tilde_observe!!(context, right, left, vname, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value and updated `vi`.

Falls back to `dot_tilde_observe!!(context, right, left, vi)` ignoring the information about variable
name and indices; if needed, these can be accessed through this function, though.
"""
function dot_tilde_observe!!(context, right, left, vn, vi)
    return dot_tilde_observe!!(context, right, left, vi)
end

"""
    dot_tilde_observe!!(context, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value and updated `vi`.

Falls back to `dot_tilde_observe(context, right, left, vi)`.
"""
function dot_tilde_observe!!(context, right, left, vi)
    logp, vi = dot_tilde_observe(context, right, left, vi)
    return left, acclogp_observe!!(context, vi, logp)
end

# Falls back to non-sampler definition.
function dot_observe(::AbstractSampler, dist, value, vi)
    return dot_observe(dist, value, vi)
end
function dot_observe(dist::MultivariateDistribution, value::AbstractMatrix, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(dist, value), vi
end
function dot_observe(dists::Distribution, value::AbstractArray, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(dists, value), vi
end
function dot_observe(dists::AbstractArray{<:Distribution}, value::AbstractArray, vi)
    increment_num_produce!(vi)
    return sum(Distributions.loglikelihood.(dists, value)), vi
end

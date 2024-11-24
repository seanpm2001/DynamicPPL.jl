function AbstractPPL.evaluate!!(
    model::Model, varinfo::AbstractVarInfo, context::AbstractContext
)
    @warn "`evaluate!!(model, args...)` is deprecated. Use `new_evaluate!!(model; kwargs...)` instead."
    return if use_threadsafe_eval(context, varinfo)
        evaluate_threadsafe!!(model, varinfo, context)
    else
        evaluate_threadunsafe!!(model, varinfo, context)
    end
end

function AbstractPPL.evaluate!!(
    model::Model,
    rng::Random.AbstractRNG,
    varinfo::AbstractVarInfo=VarInfo(),
    sampler::AbstractSampler=SampleFromPrior(),
    context::AbstractContext=DefaultContext(),
)
    return evaluate!!(model, varinfo, SamplingContext(rng, sampler, context))
end

function AbstractPPL.evaluate!!(model::Model, context::AbstractContext)
    return evaluate!!(model, VarInfo(), context)
end

function AbstractPPL.evaluate!!(
    model::Model, args::Union{AbstractVarInfo,AbstractSampler,AbstractContext}...
)
    return evaluate!!(model, Random.default_rng(), args...)
end

# without VarInfo
function AbstractPPL.evaluate!!(
    model::Model,
    rng::Random.AbstractRNG,
    sampler::AbstractSampler,
    args::AbstractContext...,
)
    return evaluate!!(model, rng, VarInfo(), sampler, args...)
end

# without VarInfo and without AbstractSampler
function AbstractPPL.evaluate!!(
    model::Model, rng::Random.AbstractRNG, context::AbstractContext
)
    return evaluate!!(model, rng, VarInfo(), SampleFromPrior(), context)
end

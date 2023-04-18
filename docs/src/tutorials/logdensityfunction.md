# Working with LogDensityProblems.jl: `LogDensityFunction`

In many cases, to define an MCMC sampler or some other approximate inference method, one really only needs a mapping of the form `θ → logπ(data, θ)` where `logπ` denotes the log joint-probability of the `data` and the parameters `θ`.

In such a case, all the intricacies and functionality provided by a [`DynamicPPL.Model`](@ref), e.g.
```@example logdensityfunction
using DynamicPPL, Distributions

@model function demo()
    m ~ Normal()
    s ~ InverseGamma()
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
end
model = demo()
nothing # hide
```

is quite redundant.

For these cases, converting a `DynamicPPL.Model` into a something that implements the much simpler LogDensityProblems.jl-interface can be very useful. In fact, [`DynamicPPL.LogDensityFunction`](@ref) does exactly this!

```@example logdensityfunction
f = DynamicPPL.LogDensityFunction(demo())
nothing # hide
```

```@example logdensityfunction
using LogDensityProblems: dimension
dimension(f)
```

```@example logdensityfunction
using LogDensityProblems: logdensity
logdensity(f, [0.0, 1.0])
```

Moreover, we can also easily take gradients using `LogDensityProblemsAD.jl`:

```@example logdensityfunction
using ForwardDiff: ForwardDiff
using LogDensityProblems: logdensity_and_gradient
using LogDensityProblemsAD: ADgradient

f_with_grad = ADgradient(Val(:ForwardDiff), f)
logdensity_and_gradient(f_with_grad, [0.0, 1.0])
```

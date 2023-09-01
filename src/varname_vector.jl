"""
    VarNameDict

A `VarNameDict` is a vector-like collection of values that can be indexed by `VarName`.

This is basically like a `OrderedDict{<:VarName}` but ensures that the underlying values
are stored contiguously in memory.
"""
struct VarNameDict{K,T,V<:AbstractVector{T},D<:AbstractDict{K}} <: AbstractDict{K,T}
    values::V
    varname_to_ranges::D
end

VarNameDict(iter) = VarNameDict(OrderedDict(iter))
function VarNameDict(dict::OrderedDict=OrderedDict{VarName,Any}())
    isempty(dict) && return VarNameDict(float(Int)[], OrderedDict{VarName,UnitRange{Int}}())

    offset = 0
    ranges = map(values(dict)) do x
        r = (offset + 1):(offset + length(x))
        offset = r[end]
        r
    end
    # TODO: Need to ensure that these are vectorized.
    vals = mapreduce(DynamicPPL.vectorize, vcat, values(dict))
    return VarNameDict(vals, OrderedDict(zip(keys(dict), ranges)))
end

# Dict-like functionality.
Base.keys(vnd::VarNameDict) = keys(vnd.varname_to_ranges)
Base.values(vnd::VarNameDict) = vnd.values
Base.length(vnd::VarNameDict) = length(vnd.values)

Base.getindex(vnd::VarNameDict, i) = getindex(vnd.values, i)
Base.setindex!(vnd::VarNameDict, val, i) = setindex!(vnd.values, val, i)

function nextrange(vnd::VarNameDict, x)
    n = length(vnd)
    return (n + 1):(n + length(x))
end

function Base.getindex(vnd::VarNameDict, vn::VarName)
    return getindex(vnd.values, vnd.varname_to_ranges[vn])
end
function Base.setindex!(vnd::VarNameDict, val, vn::VarName)
    # If we don't have `vn` in the dictionary, then we need to add it.
    if !haskey(vnd.varname_to_ranges, vn)
        # Set the range for the new variable.
        r = nextrange(vnd, val)
        vnd.varname_to_ranges[vn] = r
        # Resize the underlying vector to accommodate the new values.
        resize!(vnd.values, r[end])
    else
        # Existing keys needs to be handled differently depending on
        # whether the size of the value is increasing or decreasing.
        r = vnd.varname_to_ranges[vn]
        n_val = length(val)
        n_r = length(r)
        if n_val > n_r
            # Remove the old range.
            delete!(vnd.varname_to_ranges, vn)
            # Add the new range.
            r_new = nextrange(vnd, val)
            vnd.varname_to_ranges[vn] = r_new
            # Resize the underlying vector to accommodate the new values.
            resize!(vnd.values, r_new[end])
        else
            n_val < n_r
            # Just decrease the current range.
            vnd.varname_to_ranges[vn] = r[1]:(r[1] + n_val - 1)
        end

        # TODO: Keep track of unused ranges so we can perform sweeps
        # every now and then to free up memory and re-contiguize the
        # underlying vector.
    end

    return setindex!(vnd.values, val, vnd.varname_to_ranges[vn])
end

function BangBang.setindex!!(vnd::VarNameDict, val, vn::VarName)
    setindex!(vnd, val, vn)
    return vnd
end

function Base.iterate(vnd::VarNameDict, state=nothing)
    res = if state === nothing
        iterate(vnd.varname_to_ranges)
    else
        iterate(vnd.varname_to_ranges, state)
    end
    res === nothing && return nothing
    (vn, range), state_new = res
    return vn => vnd.values[range], state_new
end

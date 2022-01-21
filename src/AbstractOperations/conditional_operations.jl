import Oceananigans.Fields: condition_operand, conditional_length, set!

# For conditional reductions such as mean(condition_operand(u * v, 0, u > 0))

struct ConditionalOperation{LX, LY, LZ, O, G, C, M, T} <: AbstractOperation{LX, LY, LZ, G, T} 
    operand :: O
    grid :: G
    condition :: C
    mask :: M
 
     function ConditionalOperation{LX, LY, LZ}(operand::O, grid::G, condition::C, mask::M) where {LX, LY, LZ, O, G, C, M}
         T = eltype(operand)
         return new{LX, LY, LZ, O, G, C, M, T}(operand, grid, condition, mask)
     end
end
 
@inline condition_operand(operand::AbstractField, condition, mask)    = condition_operand(location(operand)..., operand.grid, condition, mask)
@inline condition_operand(LX, LY, LZ, operand, grid, condition, mask) = ConditionalOperation{LX, LY, LZ}(operand, grid, condition, mask)

@inline function evaluate_condition(c::ConditionalOperation)
    f = similar(c.operand)
    set!(f, c.condition)
    return f
end

@inline conditioned_domain(c::ConditionalOperation)       = evaluate_condition(c)
@inline conditional_length(c::ConditionalOperation)       = sum(conditioned_domain(c))
@inline conditional_length(c::ConditionalOperation, dims) = sum(f, conditioned_domain(c); dims = dims)

Adapt.adapt_structure(to, c::ConditionalOperation{LX, LY, LZ}) where {LX, LY, LZ} =
            ConditionalOperation{LX, LY, LZ}(adapt(to, c.operand), 
                                     adapt(to, c.grid),
                                     adapt(to, c.mask),
                                     adapt(to, c.condition))

@inline function Base.getindex(c::ConditionalOperation, i, j, k) 
    return ifelse(get_condition(c.condition, i, j, k, c.grid, c), 
                  getindex(c.operand, i, j, k),
                  c.mask)
end

function concretize_condition(c::ConditionalOperation)
    f = similar(c.operand)
    set!(f, c)
    return f
end

@inline get_condition(condition, i, j, k, grid, args...)                = condition(i, j, k, grid, args...)
@inline get_condition(condition::AbstractArray, i, j, k, grid, args...) = condition[i, j, k]


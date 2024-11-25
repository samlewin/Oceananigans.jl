using Oceananigans.Grids: AbstractZStarGrid
using Oceananigans.Operators

import Oceananigans.Grids: dynamic_column_depthᶜᶜᵃ, 
                           dynamic_column_depthᶜᶠᵃ,
                           dynamic_column_depthᶠᶜᵃ,
                           dynamic_column_depthᶠᶠᵃ

import Oceananigans.Operators: e₃ⁿ, e₃⁻, ∂t_e₃

const ZStarImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:AbstractZStarGrid}
const ZStarGridOfSomeKind = Union{ZStarImmersedGrid, AbstractZStarGrid}

@inline dynamic_column_depthᶜᶜᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶜᵃ(i, j, grid) +      η[i, j, grid.Nz+1]
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶜᶠᵃ(i, j, grid) +  ℑxᶠᵃᵃ(i, j, grid.Nz+1, grid, η)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶜᵃ(i, j, grid) +  ℑyᵃᶠᵃ(i, j, grid.Nz+1, grid, η)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid::ZStarGridOfSomeKind, η) = @inbounds static_column_depthᶠᶠᵃ(i, j, grid) + ℑxyᶠᶠᵃ(i, j, grid.Nz+1, grid, η)

# Convenience
@inline dynamic_column_depthᶜᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶜᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶜᶠᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶜᵃ(i, j, grid, grid.z.ηⁿ)
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid::ZStarGridOfSomeKind) = dynamic_column_depthᶠᶠᵃ(i, j, grid, grid.z.ηⁿ)

# Fallbacks
@inline e₃ⁿ(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = e₃ⁿ(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline e₃⁻(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = e₃⁻(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ∂t_e₃(i, j, k, ibg::IBG) = ∂t_e₃(i, j, k, ibg.underlying_grid)

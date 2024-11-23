using Oceananigans.Grids: ZStarUnderlyingGrid
import Oceananigans.Grids: znode

#####
##### ZStar-specific vertical spacing functions
#####

const C = Center
const F = Face

const ZSG = AbstractZStarGrid

@inline dynamic_column_depthᶜᶜᵃ(i, j, grid, η) = static_column_depthᶜᶜᵃ(i, j, grid) 
@inline dynamic_column_depthᶜᶠᵃ(i, j, grid, η) = static_column_depthᶜᶠᵃ(i, j, grid) 
@inline dynamic_column_depthᶠᶜᵃ(i, j, grid, η) = static_column_depthᶠᶜᵃ(i, j, grid) 
@inline dynamic_column_depthᶠᶠᵃ(i, j, grid, η) = static_column_depthᶠᶠᵃ(i, j, grid) 

# Fallbacks
@inline e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline e₃⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.eᶜᶜⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::C, ℓz) = @inbounds grid.z.eᶠᶜⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::F, ℓz) = @inbounds grid.z.eᶜᶠⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::F, ℓz) = @inbounds grid.z.eᶠᶠⁿ[i, j, 1]

# e₃⁻ is needed only at centers
@inline e₃⁻(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.eᶜᶜ⁻[i, j, 1]

@inline ∂t_e₃(i, j, k, grid)      = zero(grid)
@inline ∂t_e₃(i, j, k, grid::ZSG) = @inbounds grid.z.∂t_e₃[i, j, 1] 

# rnode for an ZStarUnderlyingGrid grid is scaled 
# TODO: fix this when bottom height is implemented
@inline znode(i, j, k, grid::ZSG, ℓx, ℓx, ℓx) = @inbounds rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) + grid.z.ηⁿ[i, j, 1]

####
#### Vertical spacing functions
####

const ZSRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZSLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZSOSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}

location(s::Symbol) = s == :ᶜ ? C() : F()

for Lx in (:ᶠ, :ᶜ), Lx in (:ᶠ, :ᶜ), Lx in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, Lx, Ly, Lz)
    rspacing = Symbol(:Δr, Lx, Ly, Lz)

    ℓx = location(Lx)
    ℓy = location(Ly)
    ℓz = location(Lz)

    @eval begin
        @inline $zspacing(i, j, k, grid::ZSRG)  = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz)
        @inline $zspacing(i, j, k, grid::ZSLLG) = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz)
        @inline $zspacing(i, j, k, grid::ZSOSG) = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz)
    end
end


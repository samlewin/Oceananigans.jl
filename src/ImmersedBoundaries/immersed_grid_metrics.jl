using Oceananigans.AbstractOperations: GridMetricOperation

import Oceananigans.Grids: coordinates
import Oceananigans.Operators: Δrᵃᵃᶜ, Δrᵃᵃᶠ, Δzᵃᵃᶜ, Δzᵃᵃᶠ

const c = Center()
const f = Face()
const IBG = ImmersedBoundaryGrid

# Grid metrics for ImmersedBoundaryGrid
#
# All grid metrics are defined here.
#
# For non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.
#
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    for dir in (:x, :y, :z), operator in (:Δ, :A)
    
        metric = Symbol(operator, dir, LX, LY, LZ)
        @eval begin
            import Oceananigans.Operators: $metric
            @inline $metric(i, j, k, ibg::IBG) = $metric(i, j, k, ibg.underlying_grid)
        end
    end

    metric = Symbol(:Δr, LX, LY, LZ)
    @eval begin
        import Oceananigans.Operators: $metric
        @inline $metric(i, j, k, ibg::IBG) = $metric(i, j, k, ibg.underlying_grid)
    end

    volume = Symbol(:V, LX, LY, LZ)
    @eval begin
        import Oceananigans.Operators: $volume
        @inline $volume(i, j, k, ibg::IBG) = $volume(i, j, k, ibg.underlying_grid)
    end
end

@inline Δrᵃᵃᶜ(i, j, k, ibg::IBG) = Δrᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Δrᵃᵃᶠ(i, j, k, ibg::IBG) = Δrᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)

coordinates(grid::IBG) = coordinates(grid.underlying_grid)
xspacings(X, grid::IBG) = xspacings(X, grid.underlying_grid)
yspacings(Y, grid::IBG) = yspacings(Y, grid.underlying_grid)
zspacings(Z, grid::IBG) = zspacings(Z, grid.underlying_grid)

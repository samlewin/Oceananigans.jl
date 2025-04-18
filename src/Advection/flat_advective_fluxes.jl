
#####
##### Flat Topologies
#####

using Oceananigans.Grids: XFlatGrid, YFlatGrid, ZFlatGrid

for SchemeType in [:CenteredScheme, :UpwindScheme]
    @eval begin
        @inline advective_momentum_flux_Uu(i, j, k, grid::XFlatGrid, ::$SchemeType, U, u) = zero(grid)
        @inline advective_momentum_flux_Uv(i, j, k, grid::XFlatGrid, ::$SchemeType, U, v) = zero(grid)
        @inline advective_momentum_flux_Uw(i, j, k, grid::XFlatGrid, ::$SchemeType, U, w) = zero(grid)

        @inline advective_momentum_flux_Vv(i, j, k, grid::YFlatGrid, ::$SchemeType, V, v) = zero(grid)
        @inline advective_momentum_flux_Vu(i, j, k, grid::YFlatGrid, ::$SchemeType, V, u) = zero(grid)
        @inline advective_momentum_flux_Vw(i, j, k, grid::YFlatGrid, ::$SchemeType, V, w) = zero(grid)

        @inline advective_momentum_flux_Wu(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, u) = zero(grid)
        @inline advective_momentum_flux_Wv(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, v) = zero(grid)
        @inline advective_momentum_flux_Ww(i, j, k, grid::ZFlatGrid, ::$SchemeType, W, w) = zero(grid)
    end
end

Grids = [:XFlatGrid, :YFlatGrid, :ZFlatGrid, :XFlatGrid, :YFlatGrid, :ZFlatGrid]

for (dir, Grid) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ, :xᶜᵃᵃ, :yᵃᶜᵃ, :zᵃᵃᶜ], Grids)
    bias_interp = Symbol(:biased_interpolate_, dir)
    symm_interp = Symbol(:symmetric_interpolate_, dir)
    @eval begin
        @inline $symm_interp(i, j, k, grid::$Grid, scheme, ψ, args...)           = @inbounds ψ[i, j, k]
        @inline $symm_interp(i, j, k, grid::$Grid, scheme, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)

        @inline $symm_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, ψ, args...)           = @inbounds ψ[i, j, k]
        @inline $symm_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $symm_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, ψ::Function, S::AbstractSmoothnessStencil, args...) = @inbounds ψ(i, j, k, grid, args...)
    
        @inline $bias_interp(i, j, k, grid::$Grid, scheme, bias, ψ, args...)           = @inbounds ψ[i, j, k]
        @inline $bias_interp(i, j, k, grid::$Grid, scheme, bias, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)

        @inline $bias_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, bias, ψ, args...)           = @inbounds ψ[i, j, k]
        @inline $bias_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, bias, ψ::Function, args...) = @inbounds ψ(i, j, k, grid, args...)
        @inline $bias_interp(i, j, k, grid::$Grid, scheme::AbstractUpwindBiasedAdvectionScheme, bias, ψ::Function, S::AbstractSmoothnessStencil, args...) = @inbounds ψ(i, j, k, grid, args...)
    end
end

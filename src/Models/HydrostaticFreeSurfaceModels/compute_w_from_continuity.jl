using Oceananigans.Architectures: device
using Oceananigans.Grids: halo_size, topology
using Oceananigans.Grids: XFlatGrid, YFlatGrid
using Oceananigans.Operators: flux_div_xyᶜᶜᶜ, div_xyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_cell

"""
    compute_w_from_continuity!(model)

Compute the vertical velocity ``w`` by integrating the continuity equation from the bottom upwards:

```
w^{n+1} = -∫ [∂/∂x (u^{n+1}) + ∂/∂y (v^{n+1})] dz
```
"""
compute_w_from_continuity!(model; kwargs...) =
    compute_w_from_continuity!(model.velocities, model.architecture, model.grid; kwargs...)

compute_w_from_continuity!(velocities, arch, grid; parameters = w_kernel_parameters(grid)) = 
    launch!(arch, grid, parameters, _compute_w_from_continuity!, velocities, grid)

@kernel function _compute_w_from_continuity!(U, grid)
    i, j = @index(Global, NTuple)

    @inbounds U.w[i, j, 1] = 0
    for k in 2:grid.Nz+1
        δ_Uh = flux_div_xyᶜᶜᶜ(i, j, k-1, grid, U.u, U.v) / Azᶜᶜᶜ(i, j, k-1, grid) 
        ∂t_s = Δrᶜᶜᶜ(i, j, k-1, grid) * ∂t_grid(i, j, k-1, grid)

        immersed = immersed_cell(i, j, k-1, grid)
        Δw       = δ_Uh + ifelse(immersed, 0, ∂t_s) # We do not account for grid changes in immersed cells

        @inbounds U.w[i, j, k] = U.w[i, j, k-1] - Δw
    end
end

#####
##### Size and offsets for the w kernel
#####

# extend w kernel to compute also the boundaries
# If Flat, do not calculate on halos!
@inline function w_kernel_parameters(grid) 
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    ii = ifelse(Tx == Flat, 1:Nx, -Hx+1:Nx+Hx)
    jj = ifelse(Ty == Flat, 1:Ny, -Hy+1:Ny+Hy)

    return KernelParameters(ii, jj)
end

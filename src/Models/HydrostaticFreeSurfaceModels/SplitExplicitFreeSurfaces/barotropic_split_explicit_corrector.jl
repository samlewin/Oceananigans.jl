# Kernels to compute the vertical integral of the velocities
@kernel function _barotropic_mode_kernel!(U̅, V̅, grid, ::Nothing, u, v, η)
    i, j  = @index(Global, NTuple)
    barotropic_mode_kernel!(U̅, V̅, i, j, grid, u, v, η)
end

@kernel function _barotropic_mode_kernel!(U̅, V̅, grid, active_cells_map, u, v, η)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    barotropic_mode_kernel!(U̅, V̅, i, j, grid, u, v, η)
end

@inline function barotropic_mode_kernel!(U̅, V̅, i, j, grid, u, v, η)
    k_top  = size(grid, 3) + 1

    sᶠᶜ = dynamic_column_depthᶠᶜᵃ(i, j, k_top, grid, η) / static_column_depthᶠᶜᵃ(i, j, grid)
    sᶜᶠ = dynamic_column_depthᶜᶠᵃ(i, j, k_top, grid, η) / static_column_depthᶜᶠᵃ(i, j, grid)

    @inbounds U̅[i, j, 1] = Δrᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1] * sᶠᶜ
    @inbounds V̅[i, j, 1] = Δrᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1] * sᶜᶠ

    for k in 2:grid.Nz
        @inbounds U̅[i, j, 1] += Δrᶠᶜᶜ(i, j, k, grid) * u[i, j, k] * sᶠᶜ
        @inbounds V̅[i, j, 1] += Δrᶜᶠᶜ(i, j, k, grid) * v[i, j, k] * sᶜᶠ
    end

    return nothing
end

@inline function compute_barotropic_mode!(U̅, V̅, grid, u, v, η)
    active_cells_map = retrieve_surface_active_cells_map(grid)
    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U̅, V̅, grid, active_cells_map, u, v, η; active_cells_map)
    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U, V, U̅, V̅, η, grid)
    i, j, k = @index(Global, NTuple)
    k_top = size(grid, 3) + 1

    @inbounds begin
        Hᶠᶜ = dynamic_column_depthᶠᶜᵃ(i, j, k_top, grid, η)
        Hᶜᶠ = dynamic_column_depthᶜᶠᵃ(i, j, k_top, grid, η)
        
        u[i, j, k] = u[i, j, k] + (U[i, j, 1] - U̅[i, j, 1]) / Hᶠᶜ
        v[i, j, k] = v[i, j, k] + (V[i, j, 1] - V̅[i, j, 1]) / Hᶜᶠ
    end
end

# Correcting `u` and `v` with the barotropic mode computed in `free_surface`
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    η     = free_surface.η
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U, state.V
    arch  = architecture(grid)

    # NOTE: the filtered `U̅` and `V̅` have been copied in the instantaneous `U` and `V`,
    # so we use the filtered velocities as "work arrays" to store the vertical integrals
    # of the instantaneous velocities `u` and `v`.
    compute_barotropic_mode!(U̅, V̅, grid, u, v, η)

    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U, V, U̅, V̅, η, grid)

    return nothing
end

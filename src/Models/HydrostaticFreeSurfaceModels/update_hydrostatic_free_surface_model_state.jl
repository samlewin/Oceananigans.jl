using Oceananigans.Architectures
using Oceananigans.BoundaryConditions

using Oceananigans: UpdateStateCallsite
using Oceananigans.Biogeochemistry: update_biogeochemical_state!
using Oceananigans.TurbulenceClosures: compute_diffusivities!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, mask_immersed_field_xy!, inactive_node
using Oceananigans.Models.NonhydrostaticModels: update_hydrostatic_pressure!, p_kernel_parameters
using Oceananigans.Fields: replace_horizontal_vector_halos!

import Oceananigans.TimeSteppers: update_state!
import Oceananigans.Models.NonhydrostaticModels: compute_auxiliaries!

using Oceananigans.Models: update_model_field_time_series!

compute_auxiliary_fields!(auxiliary_fields) = Tuple(compute!(a) for a in auxiliary_fields)

# Note: see single_column_model_mode.jl for a "reduced" version of update_state! for
# single column models.

"""
    update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[])

Update peripheral aspects of the model (auxiliary fields, halo regions, diffusivities,
hydrostatic pressure) to the current model state. If `callbacks` are provided (in an array),
they are called in the end.
"""
#
# BEGIN REDUCTION
#
update_state!(model::HydrostaticFreeSurfaceModel, callbacks=[]; compute_tendencies = true) =
         update_state!(prognostic_fields(model))

function update_state!(maybe_nested_tuple)

    fields = flatten_tuple(Tuple(tuplify(ai) for ai in maybe_nested_tuple))
    
    # Fill the rest
    bc = map(boundary_conditions, fields)
    
    sides = [:bottom_and_top]
    bc = Tuple((map(extract_bottom_bc, bc), map(extract_top_bc, bc)) for side in sides)

    return nothing
end

function boundary_conditions(f::Field)
    return f.boundary_conditions
end

@inline extract_bottom_bc(thing) = thing.bottom
@inline extract_top_bc(thing) = thing.top

# Utility for extracting values from nested NamedTuples
@inline tuplify(a::NamedTuple) = Tuple(tuplify(ai) for ai in a)
@inline tuplify(a) = a

# Outer-inner form
@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()

#
# END OF REDUCTION
#

# Mask immersed fields
function mask_immersed_model_fields!(model, grid)
    η = displacement(model.free_surface)
    fields_to_mask = merge(model.auxiliary_fields, prognostic_fields(model))

    foreach(fields_to_mask) do field
        if field !== η
            mask_immersed_field!(field)
        end
    end
    mask_immersed_field_xy!(η, k=size(grid, 3)+1, mask = inactive_node)

    return nothing
end

function compute_auxiliaries!(model::HydrostaticFreeSurfaceModel; w_parameters = tuple(w_kernel_parameters(model.grid)),
                                                                  p_parameters = tuple(p_kernel_parameters(model.grid)),
                                                                  κ_parameters = tuple(:xyz)) 
    
    grid = model.grid
    closure = model.closure
    diffusivity = model.diffusivity_fields

    for (wpar, ppar, κpar) in zip(w_parameters, p_parameters, κ_parameters)
        compute_w_from_continuity!(model; parameters = wpar)
        compute_diffusivities!(diffusivity, closure, model; parameters = κpar)
        update_hydrostatic_pressure!(model.pressure.pHY′, architecture(grid), 
                                    grid, model.buoyancy, model.tracers; 
                                    parameters = ppar)
    end
    return nothing
end

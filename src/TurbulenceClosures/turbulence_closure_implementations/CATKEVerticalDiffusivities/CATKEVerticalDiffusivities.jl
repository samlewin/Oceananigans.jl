module CATKEVerticalDiffusivities

using Adapt
using KernelAbstractions: @kernel, @index

using Oceananigans.Architectures
using Oceananigans.Grids
using Oceananigans.Utils
using Oceananigans.Units
using Oceananigans.Fields
using Oceananigans.Operators

using Oceananigans.Utils: prettysummary
using Oceananigans.Grids: peripheral_node, inactive_node, inactive_cell
using Oceananigans.Fields: ZeroField
using Oceananigans.BoundaryConditions: default_prognostic_bc, DefaultBoundaryCondition
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions
using Oceananigans.BoundaryConditions: DiscreteBoundaryFunction, FluxBoundaryCondition
using Oceananigans.BuoyancyModels: ∂z_b, top_buoyancy_flux

using Oceananigans.TurbulenceClosures:
    getclosure,
    time_discretization,
    AbstractScalarDiffusivity,
    VerticallyImplicitTimeDiscretization,
    VerticalFormulation

import Oceananigans.BoundaryConditions: getbc
import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures:
    validate_closure,
    shear_production,
    buoyancy_flux,
    dissipation,
    add_closure_specific_boundary_conditions,
    calculate_diffusivities!,
    DiffusivityFields,
    implicit_linear_coefficient,
    viscosity,
    diffusivity,
    viscosity_location,
    diffusivity_location,
    diffusive_flux_x,
    diffusive_flux_y,
    diffusive_flux_z

struct CATKEVerticalDiffusivity{TD, CL, FT, TKE} <: AbstractScalarDiffusivity{TD, VerticalFormulation}
    mixing_length :: CL
    turbulent_kinetic_energy_equation :: TKE
    maximum_diffusivity :: FT
    minimum_turbulent_kinetic_energy :: FT
    minimum_convective_buoyancy_flux :: FT
    negative_turbulent_kinetic_energy_damping_time_scale :: FT
end

function CATKEVerticalDiffusivity{TD}(mixing_length::CL,
                                      turbulent_kinetic_energy_equation::TKE,
                                      maximum_diffusivity::FT,
                                      minimum_turbulent_kinetic_energy::FT,
                                      minimum_convective_buoyancy_flux::FT,
                                      negative_turbulent_kinetic_energy_damping_time_scale::FT) where {TD, CL, TKE, FT}

    return CATKEVerticalDiffusivity{TD, CL, FT, TKE}(mixing_length,
                                                     turbulent_kinetic_energy_equation,
                                                     maximum_diffusivity,
                                                     minimum_turbulent_kinetic_energy,
                                                     minimum_convective_buoyancy_flux,
                                                     negative_turbulent_kinetic_energy_damping_time_scale)
end

"""
    CATKEVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(), FT=Float64;
                             mixing_length = MixingLength{FT}(),
                             turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation{FT}(),
                             maximum_diffusivity = Inf,
                             minimum_turbulent_kinetic_energy = zero(FT),
                             negative_turbulent_kinetic_energy_damping_time_scale = 1minute)

Return the `CATKEVerticalDiffusivity` turbulence closure for vertical mixing by
small-scale ocean turbulence based on the prognostic evolution of subgrid
Turbulent Kinetic Energy (TKE).

Keyword arguments
=================
  - `maximum_diffusivity`: Maximum value for tracer, momentum, and TKE diffusivities.
                           Used to clip the diffusivity when/if CATKE predicts
                           diffusivities that are too large.
                           Default: `Inf`.

  - `minimum_turbulent_kinetic_energy`: Minimum value for the turbulent kinetic energy.
                                        Can be used to model the presence "background" TKE
                                        levels due to, for example, mixing by breaking internal waves.
                                        Default: 0.

  - `negative_turbulent_kinetic_energy_damping_time_scale`: Damping time-scale for spurious negative values of TKE,
                                                            typically generated by oscillatory errors associated
                                                            with TKE advection.
                                                            Default: 1 minute.

Note that for numerical stability, it is recommended to either have a relative short
`negative_turbulent_kinetic_energy_damping_time_scale` or a reasonable
`minimum_turbulent_kinetic_energy`, or both.
"""
CATKEVerticalDiffusivity(FT::DataType; kw...) = CATKEVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

const CATKEVD{TD} = CATKEVerticalDiffusivity{TD} where TD
const CATKEVDArray{TD} = AbstractArray{<:CATKEVD{TD}} where TD
const FlavorOfCATKE{TD} = Union{CATKEVD{TD}, CATKEVDArray{TD}} where TD

include("mixing_length.jl")
include("turbulent_kinetic_energy_equation.jl")

function CATKEVerticalDiffusivity(time_discretization::TD = VerticallyImplicitTimeDiscretization(),
                                  FT = Float64;
                                  mixing_length = MixingLength(),
                                  turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation(),
                                  maximum_diffusivity = Inf,
                                  minimum_turbulent_kinetic_energy = 0,
                                  minimum_convective_buoyancy_flux = 1e-11,
                                  negative_turbulent_kinetic_energy_damping_time_scale = 1minute,
                                  warning = true) where TD

    if warning
        @warn "CATKEVerticalDiffusivity is an experimental turbulence closure that \n" *
              "is unvalidated and whose default parameters are not calibrated for \n" * 
              "realistic ocean conditions or for use in a three-dimensional \n" *
              "simulation. Use with caution and report bugs and problems with physics \n" *
              "to https://github.com/CliMA/Oceananigans.jl/issues."
    end

    mixing_length = convert_eltype(FT, mixing_length)
    turbulent_kinetic_energy_equation = convert_eltype(FT, turbulent_kinetic_energy_equation)

    return CATKEVerticalDiffusivity{TD}(mixing_length,
                                        turbulent_kinetic_energy_equation,
                                        FT(maximum_diffusivity),
                                        FT(minimum_turbulent_kinetic_energy),
                                        FT(minimum_convective_buoyancy_flux),
                                        FT(negative_turbulent_kinetic_energy_damping_time_scale))
                                  
end

function with_tracers(tracer_names, closure::FlavorOfCATKE)
    :e ∈ tracer_names ||
        throw(ArgumentError("Tracers must contain :e to represent turbulent kinetic energy " *
                            "for `CATKEVerticalDiffusivity`."))

    return closure
end

# For tuples of closures, we need to know _which_ closure is CATKE.
# Here we take a "simple" approach that sorts the tuple so CATKE is first.
# This is not sustainable though if multiple closures require this.
# The two other possibilities are:
# 1. Recursion to find which closure is CATKE in a compiler-inferrable way
# 2. Store the "CATKE index" inside CATKE via validate_closure.
validate_closure(closure_tuple::Tuple) = Tuple(sort(collect(closure_tuple), lt=catke_first))

catke_first(closure1, catke::FlavorOfCATKE) = false
catke_first(catke::FlavorOfCATKE, closure2) = true
catke_first(closure1, closure2) = false
catke_first(catke1::FlavorOfCATKE, catke2::FlavorOfCATKE) = error("Can't have two CATKEs in one closure tuple.")

#####
##### Mixing length and TKE equation
#####

@inline Riᶜᶜᶜ(i, j, k, grid, velocities, tracers, buoyancy) =
    ℑzᵃᵃᶜ(i, j, k, grid, Riᶜᶜᶠ, velocities, tracers, buoyancy)

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, tracers, buoyancy)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    S² = ∂z_u² + ∂z_v²
    Ri = N² / S²
    return ifelse(N² <= 0, zero(grid), Ri)
end

for S in (:MixingLength, :TurbulentKineticEnergyEquation)
    @eval @inline convert_eltype(::Type{FT}, s::$S) where FT = $S{FT}(; Dict(p => getproperty(s, p) for p in propertynames(s))...)
    @eval @inline convert_eltype(::Type{FT}, s::$S{FT}) where FT = s
end

#####
##### Diffusivities and diffusivity fields utilities
#####

function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfCATKE)

    default_diffusivity_bcs = (κᵘ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᶜ = FieldBoundaryConditions(grid, (Center, Center, Face)),
                               κᵉ = FieldBoundaryConditions(grid, (Center, Center, Face)))

    bcs = merge(default_diffusivity_bcs, bcs)

    κᵘ = ZFaceField(grid, boundary_conditions=bcs.κᵘ)
    κᶜ = ZFaceField(grid, boundary_conditions=bcs.κᶜ)
    κᵉ = ZFaceField(grid, boundary_conditions=bcs.κᵉ)
    Lᵉ = CenterField(grid) #, boundary_conditions=nothing)

    # Secret tuple for getting tracer diffusivities with tuple[tracer_index]
    _tupled_tracer_diffusivities         = NamedTuple(name => name === :e ? κᵉ : κᶜ          for name in tracer_names)
    _tupled_implicit_linear_coefficients = NamedTuple(name => name === :e ? Lᵉ : ZeroField() for name in tracer_names)

    return (; κᵘ, κᶜ, κᵉ, Lᵉ, _tupled_tracer_diffusivities, _tupled_implicit_linear_coefficients)
end        

const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfCATKE) = (c, c, f)
@inline diffusivity_location(::FlavorOfCATKE) = (c, c, f)

@inline clip(x) = max(zero(x), x)

function calculate_diffusivities!(diffusivities, closure::FlavorOfCATKE, model)

    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy
    clock = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    launch!(arch, grid, :xyz,
            calculate_CATKE_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)

    return nothing
end

@kernel function calculate_CATKE_diffusivities!(diffusivities, grid, closure::FlavorOfCATKE, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    i, j, k = @index(Global, NTuple)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    max_K = closure_ij.maximum_diffusivity

    @inbounds begin
        κᵘ★ = κuᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, clock, top_tracer_bcs)
        κᶜ★ = κcᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, clock, top_tracer_bcs)
        κᵉ★ = κeᶜᶜᶠ(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, clock, top_tracer_bcs)

        on_periphery = peripheral_node(i, j, k, grid, c, c, f)
        within_inactive = inactive_node(i, j, k, grid, c, c, f)
        κᵘ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, NaN, κᵘ★))
        κᶜ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, NaN, κᶜ★))
        κᵉ★ = ifelse(on_periphery, zero(grid), ifelse(within_inactive, NaN, κᵉ★))

        diffusivities.κᵘ[i, j, k] = min(max_K, κᵘ★) 
        diffusivities.κᶜ[i, j, k] = min(max_K, κᶜ★)
        diffusivities.κᵉ[i, j, k] = min(max_K, κᵉ★)

        # "Patankar trick" for buoyancy production (cf Patankar 1980 or Burchard et al. 2003)
        # If buoyancy flux is a _sink_ of TKE, we treat it implicitly.
        wb = ℑzᵃᵃᶜ(i, j, k, grid, buoyancy_fluxᶜᶜᶠ, tracers, buoyancy, diffusivities)
        eⁱʲᵏ = @inbounds tracers.e[i, j, k]

        # See `buoyancy_flux`
        dissipative_buoyancy_flux = sign(wb) * sign(eⁱʲᵏ) < 0
        wb_e = ifelse(dissipative_buoyancy_flux, wb / eⁱʲᵏ, zero(grid))

        # Implicit TKE flux at solid bottoms (extra damping for TKE near boundaries)
        on_bottom = !inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)
        Cᵂϵ = closure_ij.turbulent_kinetic_energy_equation.Cᵂϵ
        Q_e = - Cᵂϵ * turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure_ij, tracers.e) / Δz * on_bottom

        # Implicit TKE dissipation
        ϵ_e = implicit_dissipation_coefficient(i, j, k, grid, closure_ij, velocities, tracers, buoyancy, clock, top_tracer_bcs)
        
        diffusivities.Lᵉ[i, j, k] = - wb_e + ϵ_e + Q_e
    end
end

@inline function implicit_linear_coefficient(i, j, k, grid, closure::FlavorOfCATKE{<:VITD}, K, ::Val{id}, args...) where id
    L = K._tupled_implicit_linear_coefficients[id]
    return @inbounds L[i, j, k]
end

@inline function turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure, e)
    eᵢ = @inbounds e[i, j, k]
    eᵐⁱⁿ = closure.minimum_turbulent_kinetic_energy
    return sqrt(max(eᵐⁱⁿ, eᵢ))
end

@inline function κuᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓu = momentum_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓu * u★
end

@inline function κcᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓc = tracer_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓc * u★
end

@inline function κeᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    u★ = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, tracers.e)
    ℓe = TKE_mixing_lengthᶜᶜᶠ(i, j, k, grid, closure, velocities, tracers, buoyancy, clock, top_tracer_bcs)
    return ℓe * u★
end

@inline viscosity(::FlavorOfCATKE, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfCATKE, diffusivities, ::Val{id}) where id = diffusivities._tupled_tracer_diffusivities[id]
    
#####
##### Show
#####

function Base.summary(closure::CATKEVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("CATKEVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, closure::FlavorOfCATKE)
    # └
    print(io, summary(closure))
    print(io, '\n')
    print(io, "├── maximum_diffusivity: ", prettysummary(closure.maximum_diffusivity), '\n',
              "├── minimum_turbulent_kinetic_energy: ", prettysummary(closure.minimum_turbulent_kinetic_energy), '\n',
              "├── negative_turbulent_kinetic_energy_damping_time_scale: ", prettysummary(closure.negative_turbulent_kinetic_energy_damping_time_scale), '\n',
              "├── minimum_convective_buoyancy_flux: ", prettysummary(closure.minimum_convective_buoyancy_flux), '\n',
              "├── mixing_length: ", prettysummary(closure.mixing_length), '\n',
              "│   ├── Cᴺ:   ", prettysummary(closure.mixing_length.Cᴺ), '\n',
              "│   ├── Cᵇ:   ", prettysummary(closure.mixing_length.Cᵇ), '\n',
              "│   ├── Cᶜc:  ", prettysummary(closure.mixing_length.Cᶜc), '\n',
              "│   ├── Cᶜe:  ", prettysummary(closure.mixing_length.Cᶜe), '\n',
              "│   ├── Cᵉc:  ", prettysummary(closure.mixing_length.Cᵉc), '\n',
              "│   ├── Cᵉe:  ", prettysummary(closure.mixing_length.Cᵉe), '\n',
              "│   ├── C⁻u:  ", prettysummary(closure.mixing_length.C⁻u), '\n',
              "│   ├── C⁻c:  ", prettysummary(closure.mixing_length.C⁻c), '\n',
              "│   ├── C⁻e:  ", prettysummary(closure.mixing_length.C⁻e), '\n',
              "│   ├── C⁺u:  ", prettysummary(closure.mixing_length.C⁺u), '\n',
              "│   ├── C⁺c:  ", prettysummary(closure.mixing_length.C⁺c), '\n',
              "│   ├── C⁺e:  ", prettysummary(closure.mixing_length.C⁺e), '\n',
              "│   ├── CRiʷ: ", prettysummary(closure.mixing_length.CRiʷ), '\n',
              "│   └── CRiᶜ: ", prettysummary(closure.mixing_length.CRiᶜ), '\n',
              "└── turbulent_kinetic_energy_equation: ", prettysummary(closure.turbulent_kinetic_energy_equation), '\n',
              "    ├── C⁻D:  ", prettysummary(closure.turbulent_kinetic_energy_equation.C⁻D),  '\n',
              "    ├── C⁺D:  ", prettysummary(closure.turbulent_kinetic_energy_equation.C⁺D),  '\n',
              "    ├── CᶜD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᶜD),  '\n',
              "    ├── CᵉD:  ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵉD),  '\n',
              "    ├── Cᵂu★: ", prettysummary(closure.turbulent_kinetic_energy_equation.Cᵂu★), '\n',
              "    └── CᵂwΔ: ", prettysummary(closure.turbulent_kinetic_energy_equation.CᵂwΔ))
end

end # module

using Oceananigans.TimeSteppers: update_state!

import Oceananigans.Fields: set!

using Oceananigans.Utils: @apply_regionally, apply_regionally!

"""
    set!(model::HydrostaticFreeSurfaceModel; kwargs...)

Set velocity and tracer fields of `model`. The keyword arguments `kwargs...`
take the form `name = data`, where `name` refers to one of the fields of either:
(i) `model.velocities`, (ii) `model.tracers`, or (iii) `model.free_surface.η`,
and the `data` may be an array, a function with arguments `(x, y, z)`, or any data type
for which a `set!(ϕ::AbstractField, data)` function exists.

Example
=======

```jldoctest
using Oceananigans

model = HydrostaticFreeSurfaceModel(grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1)))

# Set u to a parabolic function of z, v to random numbers damped
# at top and bottom, and T to some silly array of half zeros,
# half random numbers.

u₀(x, y, z) = z / model.grid.Lz * (1 + z / model.grid.Lz)
v₀(x, y, z) = 1e-3 * rand() * u₀(x, y, z)

T₀ = rand(size(model.grid)...)
T₀[T₀ .< 0.5] .= 0

set!(model, u=u₀, v=v₀, T=T₀)

model.velocities.u

# output

16×16×16 Field{Face, Center, Center} on RectilinearGrid on CPU
├── grid: 16×16×16 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── boundary conditions: FieldBoundaryConditions
│   └── west: Periodic, east: Periodic, south: Periodic, north: Periodic, bottom: ZeroFlux, top: ZeroFlux, immersed: ZeroFlux
└── data: 22×22×22 OffsetArray(::Array{Float64, 3}, -2:19, -2:19, -2:19) with eltype Float64 with indices -2:19×-2:19×-2:19
    └── max=-0.0302734, min=-0.249023, mean=-0.166992
```
"""
@inline function set!(model::HydrostaticFreeSurfaceModel; kwargs...)
    
    maybe_nested_tuple = prognostic_fields(model)

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

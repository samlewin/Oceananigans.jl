using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation  
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarSpacingGrid
using Printf

z_faces = ZStarVerticalCoordinate(-20, 0)

grid = RectilinearGrid(size = (20, 20), 
                          x = (0, 64kilometers), 
                          z = z_faces, 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> - (64kilometers - x) / 64kilometers * 20))

model = HydrostaticFreeSurfaceModel(; grid, 
                         momentum_advection = FluxFormAdvection(WENO(; order = 5), nothing, WENO(; order = 5)),
                           tracer_advection = FluxFormAdvection(WENO(; order = 5), nothing, WENO(; order = 5)),
                                   buoyancy = BuoyancyTracer(),
                                    closure = nothing, 
                                    tracers = :b,
                               free_surface = SplitExplicitFreeSurface(; substeps = 10))

g = model.free_surface.gravitational_acceleration

model.timestepper.χ = 0.0

bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ)

Δt = 10

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 1, stop_time = 17hours) 

Δz = GridMetricOperation((Center, Center, Center), Oceananigans.AbstractOperations.Δz, model.grid)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "zstar_model") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    b  = sim.model.tracers.b
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ",  maximum(b),  minimum(b))
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

run!(simulation)

using Oceananigans.Fields: OneField

# # Check tracer conservation
b  = FieldTimeSeries("zstar_model.jld2", "b")
Δz = FieldTimeSeries("zstar_model.jld2", "Δz")

init  = sum(Δz[1] * b[1]) / sum(Δz[1])
drift = []

for t in 1:length(b.times)
  push!(drift, sum(Δz[t] * b[t]) /  sum(Δz[t]) - init) 
end


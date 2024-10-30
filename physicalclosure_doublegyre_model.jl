#using Pkg
# pkg"add Oceananigans CairoMakie"
using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation

using Oceananigans.BuoyancyModels: ∂z_b
# include("NN_closure_global.jl")
# include("xin_kai_vertical_diffusivity_local.jl")
# include("xin_kai_vertical_diffusivity_2Pr.jl")

ENV["GKSwstype"] = "100"

pushfirst!(LOAD_PATH, @__DIR__)

using Printf
using Statistics
using CairoMakie

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using SeawaterPolynomials
using SeawaterPolynomials:TEOS10
using ColorSchemes
using Glob

#%%
# filename = "doublegyre_30Cwarmflushbottom10_relaxation_30days_zC2O_CATKEVerticalDiffusivity"
filename = "doublegyre_30Cwarmflushbottom10_relaxation_30days_zWENO5_CATKEVerticalDiffusivity"
FILE_DIR = "./Output/$(filename)"
# FILE_DIR = "/storage6/xinkai/NN_Oceananigans/$(filename)"
mkpath(FILE_DIR)

# Architecture
model_architecture = GPU()

# vertical_base_closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5)
# convection_closure = XinKaiVerticalDiffusivity()
function CATKE_ocean_closure()
  mixing_length = CATKEMixingLength(Cᵇ=0.01)
  turbulent_kinetic_energy_equation = CATKEEquation(Cᵂϵ=1.0)
  return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)
end
convection_closure = CATKE_ocean_closure()
closure = convection_closure
# closure = vertical_base_closure

advection_scheme = FluxFormAdvection(WENO(order=5), WENO(order=5), WENO(order=5))

# number of grid points
const Nx = 100
const Ny = 100
const Nz = 200

const Δz = 8meters
const Lx = 4000kilometers
const Ly = 6000kilometers
const Lz = Nz * Δz

grid = RectilinearGrid(model_architecture, Float64,
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                          x = (-Lx/2, Lx/2),
                          y = (-Ly/2, Ly/2),
                          z = (-Lz, 0))

@info "Built a grid: $grid."

#####
##### Boundary conditions
#####
const T_north = 0
const T_south = 30
const T_mid = (T_north + T_south) / 2
const ΔT = T_south - T_north

const S_north = 34
const S_south = 37
const S_mid = (S_north + S_south) / 2

const τ₀ = 1e-4

const μ_drag = 1/30days
const μ_T = 1/30days

#####
##### Forcing and initial condition
#####
# @inline T_initial(x, y, z) = T_north + ΔT / 2 * (1 + z / Lz)
@inline T_initial(x, y, z) = 10 + 20 * (1 + z / Lz)

@inline surface_u_flux(x, y, t) = -τ₀ * cos(2π * y / Ly)

surface_u_flux_bc = FluxBoundaryCondition(surface_u_flux)

@inline u_drag(x, y, t, u) = @inbounds -μ_drag * Lz * u
@inline v_drag(x, y, t, v) = @inbounds -μ_drag * Lz * v

u_drag_bc  = FluxBoundaryCondition(u_drag; field_dependencies=:u)
v_drag_bc  = FluxBoundaryCondition(v_drag; field_dependencies=:v)

u_bcs = FieldBoundaryConditions(   top = surface_u_flux_bc, 
                                bottom = u_drag_bc,
                                 north = ValueBoundaryCondition(0),
                                 south = ValueBoundaryCondition(0))

v_bcs = FieldBoundaryConditions(   top = FluxBoundaryCondition(0),
                                bottom = v_drag_bc,
                                  east = ValueBoundaryCondition(0),
                                  west = ValueBoundaryCondition(0))

@inline T_ref(y) = T_mid - ΔT / Ly * y
@inline surface_T_flux(x, y, t, T) = μ_T * Δz * (T - T_ref(y))
surface_T_flux_bc = FluxBoundaryCondition(surface_T_flux; field_dependencies=:T)
T_bcs = FieldBoundaryConditions(top = surface_T_flux_bc)

@inline S_ref(y) = (S_north - S_south) / Ly * y + S_mid
@inline S_initial(x, y, z) = S_ref(y)
@inline surface_S_flux(x, y, t, S) = μ_T * Δz * (S - S_ref(y))
surface_S_flux_bc = FluxBoundaryCondition(surface_S_flux; field_dependencies=:S)
S_bcs = FieldBoundaryConditions(top = surface_S_flux_bc)

#####
##### Coriolis
#####
coriolis = BetaPlane(rotation_rate=7.292115e-5, latitude=45, radius=6371e3)

#####
##### Model building
#####

@info "Building a model..."

# This is a weird bug. If a model is not initialized with a closure other than XinKaiVerticalDiffusivity,
# the code will throw a CUDA: illegal memory access error for models larger than a small size.
# This is a workaround to initialize the model with a closure other than XinKaiVerticalDiffusivity first,
# then the code will run without any issues.
model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
    tracers = (:T, :S, :e),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    free_surface = SplitExplicitFreeSurface(grid, cfl=0.75),
    momentum_advection = advection_scheme,
    tracer_advection = advection_scheme,
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10.TEOS10EquationOfState()),
    coriolis = coriolis,
    closure = closure,
    tracers = (:T, :S, :e),
    boundary_conditions = (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs),
)

@info "Built $model."

#####
##### Initial conditions
#####

# resting initial condition
noise(z) = rand() * exp(z / 8)

T_initial_noisy(x, y, z) = T_initial(x, y, z) + 1e-6 * noise(z)
S_initial_noisy(x, y, z) = S_initial(x, y, z) + 1e-6 * noise(z)

set!(model, T=T_initial_noisy, S=S_initial_noisy)
using Oceananigans.TimeSteppers: update_state!
update_state!(model)
#####
##### Simulation building
#####
Δt₀ = 5minutes
stop_time = 10950days

simulation = Simulation(model, Δt = Δt₀, stop_time = stop_time)

# add timestep wizard callback
# wizard = TimeStepWizard(cfl=0.25, max_change=1.05, max_Δt=12minutes)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# add progress callback
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): %6.3e, max(v): %6.3e, max(T): %6.3e, max(S): %6.3e, next Δt: %s\n",
        100 * (sim.model.clock.time / sim.stop_time),
        sim.model.clock.iteration,
        prettytime(sim.model.clock.time),
        prettytime(1e-9 * (time_ns() - wall_clock[1])),
        maximum(abs, sim.model.velocities.u),
        maximum(abs, sim.model.velocities.v),
        maximum(abs, sim.model.tracers.T),
        maximum(abs, sim.model.tracers.S),
        prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

#####
##### Diagnostics
#####
u, v, w = model.velocities
T, S = model.tracers.T, model.tracers.S
U_bt = Field(Integral(u, dims=3))
Ψ = Field(CumulativeIntegral(-U_bt, dims=2))

@inline function get_N²(i, j, k, grid, b, C)
  return ∂z_b(i, j, k, grid, b, C)
end

N²_op = KernelFunctionOperation{Center, Center, Face}(get_N², model.grid, model.buoyancy.model, model.tracers)
N² = Field(N²_op)

@inline function get_density(i, j, k, grid, b, C)
  T, S = Oceananigans.BuoyancyModels.get_temperature_and_salinity(b, C)
  @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, b.model.equation_of_state)
  return ρ
end

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid, model.buoyancy, model.tracers)
ρ = Field(ρ_op)
compute!(ρ)

ρᶠ = @at (Center, Center, Face) ρ
∂ρ∂z = ∂z(ρ)
∂²ρ∂z² = ∂z(∂ρ∂z)

κc = model.diffusivity_fields.κc
wT = κc * ∂z(T)
wS = κc * ∂z(S)

ubar_zonal = Average(u, dims=1)
vbar_zonal = Average(v, dims=1)
wbar_zonal = Average(w, dims=1)
Tbar_zonal = Average(T, dims=1)
Sbar_zonal = Average(S, dims=1)
ρbar_zonal = Average(ρ, dims=1)
wTbar_zonal = Average(wT, dims=1)
wSbar_zonal = Average(wS, dims=1)

outputs = (; u, v, w, T, S, ρ, ρᶠ, ∂ρ∂z, ∂²ρ∂z², N², wT, wS)
zonal_outputs = (; ubar_zonal, vbar_zonal, wbar_zonal, Tbar_zonal, Sbar_zonal, ρbar_zonal, wTbar_zonal, wSbar_zonal)

#####
##### Build checkpointer and output writer
#####
simulation.output_writers[:xy] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xy",
                                                    indices = (:, :, Nz),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz",
                                                    indices = (1, :, :),
                                                    schedule = TimeInterval(10days))
                                                    
simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_xz",
                                                    indices = (:, 1, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_10] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_10",
                                                    indices = (10, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_20] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_20",
                                                    indices = (20, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_30] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_30",
                                                    indices = (30, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_40] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_40",
                                                    indices = (40, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_50] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_50",
                                                    indices = (50, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_60] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_60",
                                                    indices = (60, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_70] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_70",
                                                    indices = (70, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_80] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_80",
                                                    indices = (80, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:yz_90] = JLD2OutputWriter(model, outputs,
                                                    filename = "$(FILE_DIR)/instantaneous_fields_yz_90",
                                                    indices = (90, :, :),
                                                    schedule = TimeInterval(10days))

simulation.output_writers[:zonal_average] = JLD2OutputWriter(model, zonal_outputs,
                                                             filename = "$(FILE_DIR)/averaged_fields_zonal",
                                                             schedule = TimeInterval(10days))

simulation.output_writers[:streamfunction] = JLD2OutputWriter(model, (; Ψ=Ψ,),
                                                    filename = "$(FILE_DIR)/averaged_fields_streamfunction",
                                                    schedule = AveragedTimeInterval(1825days, window=1825days))

simulation.output_writers[:checkpointer] = Checkpointer(model,
                                                    schedule = TimeInterval(730days),
                                                    prefix = "$(FILE_DIR)/checkpointer")

@info "Running the simulation..."

try
  files = readdir(FILE_DIR)
  checkpoint_files = files[occursin.("checkpointer_iteration", files)]
  if !isempty(checkpoint_files)
      checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
      pickup_iter = maximum(checkpoint_iters)
      run!(simulation, pickup="$(FILE_DIR)/checkpointer_iteration$(pickup_iter).jld2")
  else
      run!(simulation)
  end
catch err
  @info "run! threw an error! The error message is"
  showerror(stdout, err)
end

#%%
T_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "T")
T_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "T")
T_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "T")

S_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "S")
S_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "S")
S_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "S")

u_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "u")
u_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "u")
u_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "u")

v_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "v")
v_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "v")
v_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "v")

times = T_xy_data.times ./ 24 ./ 60^2
Nt = length(times)
timeframes = 1:Nt

# Nx, Ny, Nz = T_xy_data.grid.Nx, T_xy_data.grid.Ny, T_xy_data.grid.Nz
xC, yC, zC = T_xy_data.grid.xᶜᵃᵃ[1:Nx], T_xy_data.grid.yᵃᶜᵃ[1:Ny], T_xy_data.grid.zᵃᵃᶜ[1:Nz]
zF = T_xy_data.grid.zᵃᵃᶠ[1:Nz+1]

# Lx, Ly, Lz = T_xy_data.grid.Lx, T_xy_data.grid.Ly, T_xy_data.grid.Lz

xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
for j in axes(zCs_yz, 2)
  zCs_yz[:, j] .= zC
end

xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
for i in axes(zCs_xz, 1)
  zCs_xz[i, :] .= zC
end

xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
for j in axes(zFs_yz, 2)
  zFs_yz[:, j] .= zF
end

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
for i in axes(zFs_xz, 1)
  zFs_xz[i, :] .= zF
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

# for freeconvection
# startheight = 64

# for wind mixing
startheight = 1
Tlim = (find_min(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(T_xy_data, :, :, 1, timeframes), interior(T_yz_data, 1, :, startheight:Nz, timeframes), interior(T_xz_data, :, 1, startheight:Nz, timeframes)))
Slim = (find_min(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)), 
        find_max(interior(S_xy_data, :, :, 1, timeframes), interior(S_yz_data, 1, :, startheight:Nz, timeframes), interior(S_xz_data, :, 1, startheight:Nz, timeframes)))
ulim = (-find_max(interior(u_xy_data, :, :, 1, timeframes), interior(u_yz_data, 1, :, startheight:Nz, timeframes), interior(u_xz_data, :, 1, startheight:Nz, timeframes)),
         find_max(interior(u_xy_data, :, :, 1, timeframes), interior(u_yz_data, 1, :, startheight:Nz, timeframes), interior(u_xz_data, :, 1, startheight:Nz, timeframes)))
vlim = (-find_max(interior(v_xy_data, :, :, 1, timeframes), interior(v_yz_data, 1, :, startheight:Nz, timeframes), interior(v_xz_data, :, 1, startheight:Nz, timeframes)),
         find_max(interior(v_xy_data, :, :, 1, timeframes), interior(v_yz_data, 1, :, startheight:Nz, timeframes), interior(v_xz_data, :, 1, startheight:Nz, timeframes)))

colorscheme = colorschemes[:balance]
T_colormap = colorscheme
S_colormap = colorscheme
u_colormap = colorscheme
v_colormap = colorscheme

T_color_range = Tlim
S_color_range = Slim
u_color_range = ulim
v_color_range = vlim
#%%
plot_aspect = (2, 3, 0.5)
fig = Figure(size=(1500, 700))
axT = Axis3(fig[1, 1], title="Temperature (°C)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axS = Axis3(fig[1, 3], title="Salinity (g kg⁻¹)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axu = Axis3(fig[2, 1], title="u (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)
axv = Axis3(fig[2, 3], title="v (m/s)", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect)

n = Observable(1)

T_xy = @lift interior(T_xy_data[$n], :, :, 1)
T_yz = @lift transpose(interior(T_yz_data[$n], 1, :, :))
T_xz = @lift interior(T_xz_data[$n], :, 1, :)

S_xy = @lift interior(S_xy_data[$n], :, :, 1)
S_yz = @lift transpose(interior(S_yz_data[$n], 1, :, :))
S_xz = @lift interior(S_xz_data[$n], :, 1, :)

u_xy = @lift interior(u_xy_data[$n], :, :, 1)
u_yz = @lift transpose(interior(u_yz_data[$n], 1, :, :))
u_xz = @lift interior(u_xz_data[$n], :, 1, :)

v_xy = @lift interior(v_xy_data[$n], :, :, 1)
v_yz = @lift transpose(interior(v_yz_data[$n], 1, :, :))
v_xz = @lift interior(v_xz_data[$n], :, 1, :)

# time_str = @lift "Surface Cooling, Time = $(round(times[$n], digits=2)) hours"
time_str = @lift "Surface Wind Stress, Time = $(round(times[$n], digits=2)) days"
Label(fig[0, :], text=time_str, tellwidth=false, font=:bold)

T_xy_surface = surface!(axT, xCs_xy, yCs_xy, zCs_xy, color=T_xy, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
T_yz_surface = surface!(axT, xCs_yz, yCs_yz, zCs_yz, color=T_yz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])
T_xz_surface = surface!(axT, xCs_xz, yCs_xz, zCs_xz, color=T_xz, colormap=T_colormap, colorrange = T_color_range, lowclip=T_colormap[1])

S_xy_surface = surface!(axS, xCs_xy, yCs_xy, zCs_xy, color=S_xy, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
S_yz_surface = surface!(axS, xCs_yz, yCs_yz, zCs_yz, color=S_yz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])
S_xz_surface = surface!(axS, xCs_xz, yCs_xz, zCs_xz, color=S_xz, colormap=S_colormap, colorrange = S_color_range, lowclip=S_colormap[1])

u_xy_surface = surface!(axu, xCs_xy, yCs_xy, zCs_xy, color=u_xy, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])
u_yz_surface = surface!(axu, xCs_yz, yCs_yz, zCs_yz, color=u_yz, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])
u_xz_surface = surface!(axu, xCs_xz, yCs_xz, zCs_xz, color=u_xz, colormap=u_colormap, colorrange = u_color_range, lowclip=u_colormap[1], highclip=u_colormap[end])

v_xy_surface = surface!(axv, xCs_xy, yCs_xy, zCs_xy, color=v_xy, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])
v_yz_surface = surface!(axv, xCs_yz, yCs_yz, zCs_yz, color=v_yz, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])
v_xz_surface = surface!(axv, xCs_xz, yCs_xz, zCs_xz, color=v_xz, colormap=v_colormap, colorrange = v_color_range, lowclip=v_colormap[1], highclip=v_colormap[end])

Colorbar(fig[1,2], T_xy_surface)
Colorbar(fig[1,4], S_xy_surface)
Colorbar(fig[2,2], u_xy_surface)
Colorbar(fig[2,4], v_xy_surface)

xlims!(axT, (-Lx/2, Lx/2))
xlims!(axS, (-Lx/2, Lx/2))
xlims!(axu, (-Lx/2, Lx/2))
xlims!(axv, (-Lx/2, Lx/2))

ylims!(axT, (-Ly/2, Ly/2))
ylims!(axS, (-Ly/2, Ly/2))
ylims!(axu, (-Ly/2, Ly/2))
ylims!(axv, (-Ly/2, Ly/2))

zlims!(axT, (-Lz, 0))
zlims!(axS, (-Lz, 0))
zlims!(axu, (-Lz, 0))
zlims!(axv, (-Lz, 0))

@info "Recording 3D fields"
CairoMakie.record(fig, "$(FILE_DIR)/$(filename)_3D_instantaneous_fields.mp4", 1:Nt, framerate=20, px_per_unit=2) do nn
    n[] = nn
end

@info "Done!"
#%%
# Ψ_data = FieldTimeSeries("$(FILE_DIR)/averaged_fields_streamfunction.jld2", "Ψ")

# xF = Ψ_data.grid.xᶠᵃᵃ[1:Ψ_data.grid.Nx+1]
# yC = Ψ_data.grid.yᵃᶜᵃ[1:Ψ_data.grid.Ny]

# Nt = length(Ψ_data)
# times = Ψ_data.times / 24 / 60^2 / 365
# #%%
# timeframe = Nt
# Ψ_frame = interior(Ψ_data[timeframe], :, :, 1) ./ 1e6
# clim = maximum(abs, Ψ_frame) + 1e-13
# N_levels = 16
# levels = range(-clim, stop=clim, length=N_levels)
# fig = Figure(size=(800, 800))
# ax = Axis(fig[1, 1], xlabel="x (m)", ylabel="y (m)", title="CATKE Vertical Diffusivity, Yearly-Averaged Barotropic streamfunction Ψ, Year $(times[timeframe])")
# cf = contourf!(ax, xF, yC, Ψ_frame, levels=levels, colormap=Reverse(:RdBu_11))
# Colorbar(fig[1, 2], cf, label="Ψ (Sv)")
# tightlimits!(ax)
# save("$(FILE_DIR)/barotropic_streamfunction_$(timeframe).png", fig, px_per_unit=4)
# display(fig)
#%%
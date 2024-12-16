using CairoMakie, JLD2, Statistics, HDF5, Oceananigans, ProgressBars
using KernelAbstractions
using Oceananigans.Utils
using Oceananigans.Architectures: device
using KernelAbstractions: @kernel, @index
using Oceananigans.Operators
using Oceananigans.Architectures: architecture

include("coarse_graining_utils.jl")

plot_data = true 
save_data = true
plot_stream_function = true
data_directory = "/orcd/data/raffaele/001/sandre/OceananigansData/"
analysis_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/"
figure_directory = "oceananigans_figure/"
figure_directory = "quick_check_figure/"

casevar = 5

si = 1000 #starting index
levels = 1:1
kmax = 2
NN = 3

jlfile = jldopen(data_directory * "baroclinic_double_gyre_free_surface_$casevar.jld2", "r")
ηkeys =  keys(jlfile["timeseries"]["η"])[2:end]

η = zeros(size(jlfile["timeseries"]["η"]["0"])[1:2]..., length(ηkeys))

for (i, ηkey) in ProgressBar(enumerate(ηkeys))
    η[:, :, i] .= jlfile["timeseries"]["η"][ηkey]
end
close(jlfile)
@info "closing jld2 file"


month_indices = zeros(Int, NN^2)
if length(ηkeys) ≥ 5000
    month_indices[1] = 1
    month_indices[2] = argmax( squareheight )
    month_indices[3] = 100 
    month_indices[4] = 1000
    month_indices[5] = 2000
    month_indices[6] = 3000
    month_indices[7] = 4000
    month_indices[8] = 5000
    month_indices[9] = 5100
else
    month_indices .= rand(1:length(ηkeys), NN^2)
end

if plot_data 
    @info "plotting data"
    etamax = maximum(abs.(η[:, :, end]))
    fig = Figure() 
    for i in 1:NN
        for j in 1:NN
            ii = (i - 1) * NN + j
            ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
            heatmap!(ax, η[:, :, month_indices[ii]], colormap = :balance, colorrange = (-etamax, etamax))
        end
    end
    save(figure_directory * "etafield.png", fig)

    @info "plotting coarse-grained data"
    for factor in [2, 4, 8, 16, 32, 64, 128]
        etamax = maximum(abs.(η[:, :, end]))
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
                heatmap!(ax, coarse_grain(η[:, :, month_indices[ii]], factor), colormap = :balance, colorrange = (-etamax, etamax))
            end
        end
        save(figure_directory * "coarsegrained_factor_$(factor)_etafield.png", fig)
    end

    squareheight = [mean(η[:, :, i] .^2) for i in eachindex(ηkeys)]


    fig = Figure()
    ax = Axis(fig[1,1]; xlabel = "time", ylabel = "mean(η^2)")
    lines!(ax, squareheight, color = :blue)
    save(figure_directory * "squareheight.png", fig)
end


u = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "u"; backend = InMemory(225))
v = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "v"; backend = InMemory(225))
w = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "w"; backend = InMemory(225))
b = FieldTimeSeries(data_directory  * "baroclinic_double_gyre_$casevar.jld2", "b"; backend = InMemory(225))

zs = u.grid.zᵃᵃᶜ[1:15]


meanabsus = zeros(15, length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanabsus[:, i] = mean(abs.(interior(u[i])), dims = (1, 2))
end
meanabsvs = zeros(15, length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanabsvs[:, i] = mean(abs.(interior(v[i])), dims = (1, 2))
end
meanabsws = zeros(15, length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    w1 =  mean(abs.(interior(w[i])), dims = (1, 2))[:]
    meanabsws[:, i] .= (w1[1:end-1] + w1[2:end]) / 2
end
meanbs = zeros(15, length(ηkeys))
for i in ProgressBar(1:length(ηkeys))
    meanbs[:, i] .= mean(interior(b[i]), dims = (1, 2))[:]
end
if plot_data
    for plot_index in 1:15
        fig = Figure()
        ax = Axis(fig[1,1]; xlabel = "time", ylabel = "level $plot_index mean(b)")
        lines!(ax, meanbs[plot_index, :], color = :blue)
        ax = Axis(fig[1,2]; xlabel = "time", ylabel = "level $plot_index mean(|u|)")
        lines!(ax, meanabsus[plot_index, :], color = :blue)
        ax = Axis(fig[2, 1]; xlabel = "time", ylabel = "level $plot_index mean(|v|)")
        lines!(ax, meanabsvs[plot_index, :], color = :blue)
        ax = Axis(fig[2, 2]; xlabel = "time", ylabel = "level $plot_index mean(|w|)")
        lines!(ax, meanabsws[plot_index, :], color = :blue)
        save(figure_directory * "level_$(plot_index)_means.png", fig)
    end
end


##
meanbs_avgd_8 = zeros(32, 32, 15, length(ηkeys));
for i in ProgressBar(1:length(ηkeys))
    bfield = interior(b[i])
    for k in 1:15
        meanbs_avgd_8[:, :, k, i] .= coarse_grain(bfield[:, :, k], 8)
    end
end

if plot_data 
    @info "plotting data"
    for plot_index in 1:15
        blims = extrema(meanbs_avgd_8[:, :, plot_index, (length(ηkeys)÷4 * 3):end])
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
                heatmap!(ax, meanbs_avgd_8[:, :, plot_index, month_indices[ii]], colormap = :thermometer, colorrange = blims)
            end
        end
        save(figure_directory * "relative_bfield_level_$(plot_index).png", fig)
    end

    blims = extrema(meanbs_avgd_8[:, :, :, (length(ηkeys)÷4 * 3):end])
    for plot_index in 1:15
        fig = Figure() 
        for i in 1:NN
            for j in 1:NN
                ii = (i - 1) * NN + j
                ax = Axis(fig[i, j]; xlabel = "x", ylabel = "y", title = "$(month_indices[ii])")
                heatmap!(ax, meanbs_avgd_8[:, :, plot_index, month_indices[ii]], colormap = :thermometer, colorrange = blims)
            end
        end
        save(figure_directory * "absolute_bfield_level_$(plot_index).png", fig)
    end
end

fig = Figure(resolution = (2000, 2000)) 
field = meanbs_avgd_8;
months = Int[]

ij_index = (16, 16)
ij_indices = [(i, j) for i in 4:4:20, j in 4:4:20]
for (k, ij_index) in enumerate(ij_indices)
    ii = (k - 1) ÷ 5 + 1
    jj = (k - 1) % 5 + 1
    i_index = ij_index[1]
    j_index = ij_index[2]
    ax = Axis(fig[6 - ii, jj]; ylabel = "depth", xlabel = "buoyancy", title = "location $(i_index), $(j_index)")
    month_color = [:black, :red, :purple, :orange, :green, :blue]
    for (i, mi) in enumerate([120, 1000, 2000, 3000, 4000, 5000])
        if k == 1
            push!(months, mi)
        end
        field_2std = 2 * std(field[i_index, j_index, :, mi-12:mi+12], dims = 2)[:]
        mfield = mean(field[i_index, j_index, :, mi-12:mi+12], dims = 2)[:]
        scatter!(ax, mfield, zs,  color = month_color[i])
        lines!(ax, mfield, zs,  color = month_color[i], label = string(mi))
    end
    if k ==1
        axislegend(ax, position = :rb)
    end
end
save(figure_directory * "buoyancy_profile.png", fig)

#=
mi = month_indices[end-3]
push!(months, mi)
mfield = mean(field[i_index, j_index, :, mi-23:mi], dims = 2)[:]
scatter!(ax, mfield, zs,  color = :red)
lines!(ax, mfield, zs,  color = :red, label = string(mi))

mi = month_indices[end-4]
push!(months, mi)
mfield = mean(field[i_index, j_index, :, mi-23:mi], dims = 2)[:]
scatter!(ax, mfield, zs,  color = :purple)
lines!(ax, mfield, zs,  color = :purple, label = string(mi))

mi = month_indices[end-6]
push!(months, mi)
mfield = mean(field[i_index, j_index, :, mi-23:mi], dims = 2)[:]
scatter!(ax, mfield, zs,  color = :orange)
lines!(ax, mfield, zs,  color = :orange, label = string(mi))
=#

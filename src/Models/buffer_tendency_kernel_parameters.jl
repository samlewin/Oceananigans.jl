function buffer_tendency_kernel_parameters(grid, arch)
    Nx, Ny, Nz = size(grid)
    Hx, Hy, _  = halo_size(grid)

    param_west  = (1:Hx,       1:Ny,       1:Nz)
    param_east  = (Nx-Hx+1:Nx, 1:Ny,       1:Nz)
    param_south = (1:Nx,       1:Hy,       1:Nz)
    param_north = (1:Nx,       Ny-Hy+1:Ny, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# p needs computing in the range  0 : 0 and N + 1 : N + 1
function buffer_p_kernel_parameters(grid, arch)
    Nx, Ny, _ = size(grid)

    param_west  = (0:0,       1:Ny)
    param_east  = (Nx+1:Nx+1, 1:Ny)
    param_south = (1:Nx,      0:0)
    param_north = (1:Nx,      Ny+1:Ny+1)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# diffusivities need recomputing in the range 0 : B and N - B + 1 : N + 1
function buffer_κ_kernel_parameters(grid, closure, arch)
    Nx, Ny, Nz = size(grid)

    Bx = required_halo_size_x(closure)
    By = required_halo_size_y(closure)

    param_west  = (0:Bx,         1:Ny,         1:Nz)
    param_east  = (Nx-Bx+1:Nx+1, 1:Ny,         1:Nz)
    param_south = (1:Nx,         0:By,         1:Nz)
    param_north = (1:Nx,         Ny-By+1:Ny+1, 1:Nz)

    params = (param_west, param_east, param_south, param_north)
    return buffer_parameters(params, grid, arch)
end

# Recompute only on communicating sides 
function buffer_parameters(parameters, grid, arch) 
    Rx, Ry, _ = arch.ranks
    Tx, Ty, _ = topology(grid)

    include_west  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == RightConnected)
    include_east  = !isa(grid, XFlatGrid) && (Rx != 1) && !(Tx == LeftConnected)
    include_south = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == RightConnected)
    include_north = !isa(grid, YFlatGrid) && (Ry != 1) && !(Ty == LeftConnected)

    include_side = (include_west, include_east, include_south, include_north)

    return Tuple(KernelParameters(parameters[i]...) for i in findall(include_side))
end




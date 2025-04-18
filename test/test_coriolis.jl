include("dependencies_for_runtests.jl")

using Oceananigans.Coriolis: Ω_Earth
using Oceananigans.Advection: EnergyConserving, EnstrophyConserving

test_fplane(::Nothing) = FPlane(f=π)
test_fplane(FT)        = FPlane(FT, f=π)
test_bplane(::Nothing) = BetaPlane(f₀=π, β=2π)
test_bplane(FT)        = BetaPlane(FT, f₀=π, β=2π)
test_ccc(::Nothing)    = ConstantCartesianCoriolis(f=1, rotation_axis=[0, cosd(45), sind(45)])
test_ccc(FT)           = ConstantCartesianCoriolis(FT, f=1, rotation_axis=[0, cosd(45), sind(45)])
test_hsc(::Nothing)    = HydrostaticSphericalCoriolis(scheme=EnergyConserving())
test_hsc(FT)           = HydrostaticSphericalCoriolis(FT, scheme=EnergyConserving(FT))
test_hsc2(::Nothing)   = HydrostaticSphericalCoriolis(rotation_rate=π)
test_hsc2(FT)          = HydrostaticSphericalCoriolis(FT, rotation_rate=π)
test_ntbp(::Nothing)   = NonTraditionalBetaPlane(rotation_rate=π, latitude=17, radius=ℯ)
test_ntbp(FT)          = NonTraditionalBetaPlane(FT, rotation_rate=π, latitude=17, radius=ℯ)

function instantiate_fplane_1(FT)
    coriolis = FPlane(FT, f=π)
    return coriolis.f ≈ FT(π)
end

function instantiate_fplane_2(FT)
    coriolis = FPlane(FT, rotation_rate=2, latitude=30)
    return coriolis.f ≈ FT(2)
end

function instantiate_constant_coriolis_1(FT)
    coriolis = ConstantCartesianCoriolis(FT, f=1, rotation_axis=[0, cosd(45), sind(45)])
    @test coriolis.fy ≈ FT(cosd(45))
    @test coriolis.fz ≈ FT(sind(45))
end

function instantiate_constant_coriolis_2(FT)
    coriolis = ConstantCartesianCoriolis(FT, f=10, rotation_axis=[√(1/3),√(1/3),√(1/3)])
    @test coriolis.fx ≈ FT(10 * √(1/3))
    @test coriolis.fy ≈ FT(10 * √(1/3))
    @test coriolis.fz ≈ FT(10 * √(1/3))
end

function instantiate_betaplane_1(FT)
    coriolis = BetaPlane(FT, f₀=π, β=2π)
    @test coriolis.f₀ ≈ FT(π)
    @test coriolis.β  ≈ FT(2π)
end

function instantiate_betaplane_2(FT)
    coriolis = BetaPlane(FT, latitude=70, radius=2π, rotation_rate=3π)
    @test coriolis.f₀ ≈ FT(6π * sind(70))
    @test coriolis.β  ≈ FT(6π * cosd(70) / 2π)
end

function instantiate_ntbetaplane_1(FT)
    coriolis = NonTraditionalBetaPlane(FT, fz=π, fy=ℯ, β=1//7, γ=5)
    @test coriolis.fz ≈ FT(π)
    @test coriolis.fy ≈ FT(ℯ)
    @test coriolis.β  ≈ FT(1//7)
    @test coriolis.γ  ≈ FT(5)
end

function instantiate_ntbetaplane_2(FT)
    Ω, φ, R = π, 17, ℯ
    coriolis = NonTraditionalBetaPlane(FT, rotation_rate=Ω, latitude=φ, radius=R)
    @test coriolis.fz ≈ FT(+ 2Ω * sind(φ))
    @test coriolis.fy ≈ FT(+ 2Ω * cosd(φ))
    @test coriolis.β  ≈ FT(+ 2Ω * cosd(φ) / R)
    @test coriolis.γ  ≈ FT(- 4Ω * sind(φ) / R)
end

function instantiate_hydrostatic_spherical_coriolis1(FT)
    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnergyConserving())
    @test coriolis.rotation_rate == FT(Ω_Earth) # default
    @test coriolis.scheme isa EnergyConserving

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    @test coriolis.rotation_rate == FT(Ω_Earth) # default
    @test coriolis.scheme isa EnstrophyConserving
end

function instantiate_hydrostatic_spherical_coriolis2(FT)
    coriolis = HydrostaticSphericalCoriolis(FT, rotation_rate=π)
    @test coriolis.rotation_rate == FT(π)
    @test coriolis.scheme isa EnstrophyConserving # default
end

@testset "Coriolis" begin
    @info "Testing Coriolis..."
    # Save for later use
    FT₀ = Oceananigans.defaults.FloatType

    for FT in float_types
        @test instantiate_fplane_1(FT)
        @test instantiate_fplane_2(FT)

        instantiate_constant_coriolis_1(FT)
        instantiate_constant_coriolis_2(FT)
        instantiate_betaplane_1(FT)
        instantiate_betaplane_2(FT)
        instantiate_hydrostatic_spherical_coriolis1(FT)
        instantiate_hydrostatic_spherical_coriolis2(FT)

        # Test that FPlane throws an ArgumentError
        @test_throws ArgumentError FPlane(FT)
        @test_throws ArgumentError FPlane(FT, rotation_rate=7e-5)
        @test_throws ArgumentError FPlane(FT, f=1, latitude=40)
        @test_throws ArgumentError FPlane(FT, f=1, rotation_rate=7e-5, latitude=40)

        # Test that ConstantCartesianCoriolis throws an ArgumentError
        @test_throws ArgumentError ConstantCartesianCoriolis(FT, rotation_axis=[0,1,1])
        @test_throws ArgumentError ConstantCartesianCoriolis(FT, f=1, latitude=45,)
        @test_throws ArgumentError ConstantCartesianCoriolis(FT, fx=1, latitude=45)
        @test_throws ArgumentError ConstantCartesianCoriolis(FT, fx=1, f=1)

        # Non-exhaustively test that BetaPlane throws an ArgumentError
        @test_throws ArgumentError BetaPlane(FT)
        @test_throws ArgumentError BetaPlane(FT, f₀=1)
        @test_throws ArgumentError BetaPlane(FT, β=1)
        @test_throws ArgumentError BetaPlane(FT, f₀=1e-4, β=1e-11, latitude=70)

        # Test that NonTraditionalBetaPlane throws an ArgumentError
        @test_throws ArgumentError NonTraditionalBetaPlane(FT)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, rotation_rate=7e-5)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, rotation_rate=7e-5, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fy=1, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fy=1, rotation_rate=7e-5, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, rotation_rate=7e-5, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, β=3, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, β=3, rotation_rate=7e-5, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, β=3, γ=4, latitude=40)
        @test_throws ArgumentError NonTraditionalBetaPlane(FT, fz=1, fy=2, β=3, γ=4, rotation_rate=7e-5, latitude=40)

        # Test show functions
        ✈ = FPlane(FT, latitude=45)
        show(✈); println()
        @test ✈ isa FPlane{FT}

        ✈ = ConstantCartesianCoriolis(FT, f=1e-4)
        show(✈); println()
        @test ✈ isa ConstantCartesianCoriolis{FT}

        ✈ = BetaPlane(FT, latitude=45)
        show(✈); println()
        @test ✈ isa BetaPlane{FT}

        ✈ = NonTraditionalBetaPlane(FT, latitude=45)
        show(✈); println()
        @test ✈ isa NonTraditionalBetaPlane{FT}

        for make_test_coriolis in (test_fplane,
                                    test_bplane,
                                    test_ccc,
                                    test_hsc,
                                    test_hsc2,
                                    test_ntbp)

            Oceananigans.defaults.FloatType = FT
            c_with_default = make_test_coriolis(nothing)
            Oceananigans.defaults.FloatType = FT₀

            c_explicitly = make_test_coriolis(FT)
            @test c_with_default === c_explicitly
        end
    end
end

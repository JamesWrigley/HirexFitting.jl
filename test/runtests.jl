using Test
using DimensionalData
using HirexFitting: HirexFitting as HF, gaussian, double_gaussian, fit_gaussian

@testset "DimVector lookup is used as x data" begin
    energies = range(9000.0, step=0.5, length=200)
    y = DimArray([gaussian(x, 1.0, 50.0, 9040.0, 5.0) for x in energies], X(energies))

    @test fit_gaussian(y) ≈ [1.0, 50.0, 9040.0, 5.0] rtol=1e-6
    @test fit_gaussian(y; workspace=HF.FitWorkspace(y)) ≈ [1.0, 50.0, 9040.0, 5.0] rtol=1e-6

    # A workspace made from a plain vector still works with a non-integer lookup
    @test fit_gaussian(y; workspace=HF.FitWorkspace(parent(y))) ≈ [1.0, 50.0, 9040.0, 5.0] rtol=1e-6

    # No lookup: fall back to indices
    y_nolookup = DimArray(parent(y), X(DimensionalData.NoLookup()))
    @test fit_gaussian(y_nolookup)[3] ≈ 81.0 rtol=1e-6
end

@testset "double_gaussian" begin
    # Narrow seeding peak on top of a broad SASE peak. Fits must be done in Float64.
    seeding = (400.0, 300.0, 3.0)
    sase = (150.0, 350.0, 80.0)
    xs = 1:731
    y = [double_gaussian(x, seeding..., sase...) for x in xs]

    params = fit_gaussian(y; f=double_gaussian)
    @test params ≈ [seeding..., sase...] rtol=1e-3

    # Check the struct output, field by field
    params = fit_gaussian(y; f=double_gaussian, params_as_array=false)
    @test params isa HF.DoubleGaussianParams
    for (fitted, expected) in ((params.seeding, seeding), (params.sase, sase))
        @test fitted.y0 == 0
        @test fitted.A ≈ expected[1] rtol=1e-3
        @test fitted.mu ≈ expected[2] rtol=1e-3
        @test fitted.sigma ≈ expected[3] rtol=1e-3
    end
end

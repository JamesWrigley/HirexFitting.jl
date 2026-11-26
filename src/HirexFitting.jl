module HirexFitting

import MKL
import Peaks
using DimensionalData
using NaNStatistics: nanmean, movmean!
import OhMyThreads as OMT
import PrecompileTools: @compile_workload
using ADTypes: AutoForwardDiff
using CurveFit: CurveFit, NonlinearCurveFitProblem, NonlinearFunction,
    CurveFitSolution, NonlinearSolveFirstOrder as NLS, ScalarModel, solve, solve!


struct GaussianParams{T}
    y0::T
    A::T
    mu::T
    sigma::T

    function GaussianParams(y0, A, mu, sigma)
        T = promote_type(typeof(y0), typeof(A), typeof(mu), typeof(sigma), Float32)
        new{T}(T(y0), T(A), T(mu), T(sigma))
    end
end

function GaussianParams(params::AbstractVector)
    n_params = length(params)
    if n_params == 4
        GaussianParams(params[1], params[2], params[3], params[4])
    elseif n_params == 3
        GaussianParams(0, params[1], params[2], params[3])
    else
        throw(ArgumentError("Params vector must have either 3 or 4 elements, not $(n_params)"))
    end
end

gaussian_nobaseline(x, A, μ, σ) = A * exp(-(x - μ)^2 / (2 * σ^2))
gaussian_nobaseline(x, p::GaussianParams) = gaussian_nobaseline(x, p.A, p.mu, p.sigma)
gaussian(x, y0, A, μ, σ) = gaussian_nobaseline(x, A, μ, σ) + y0
gaussian(x, p::GaussianParams) = gaussian(x, p.y0, p.A, p.mu, p.sigma)

"""
    double_gaussian(x, A_seeding, μ_seeding, σ_seeding, A_sase, μ_sase, σ_sase)

Sum of two Gaussians without a baseline: a narrow seeding peak and a broad SASE peak.
After fitting, the narrower of the two components is always reported as the seeding
peak.
"""
double_gaussian(x, A₁, μ₁, σ₁, A₂, μ₂, σ₂) = gaussian_nobaseline(x, A₁, μ₁, σ₁) + gaussian_nobaseline(x, A₂, μ₂, σ₂)

struct DoubleGaussianParams{T}
    seeding::GaussianParams{T}
    sase::GaussianParams{T}
end

function DoubleGaussianParams(params::AbstractVector)
    if length(params) != 6
        throw(ArgumentError("Params vector must have 6 elements, not $(length(params))"))
    end

    seeding = GaussianParams(0, params[1], params[2], params[3])
    sase = GaussianParams(0, params[4], params[5], params[6])
    T = promote_type(typeof(seeding.A), typeof(sase.A))
    DoubleGaussianParams{T}(seeding, sase)
end

double_gaussian(x, p::DoubleGaussianParams) = gaussian_nobaseline(x, p.seeding) + gaussian_nobaseline(x, p.sase)

parameter_count(::typeof(gaussian)) = 4
parameter_count(::typeof(gaussian_nobaseline)) = 3
parameter_count(::typeof(double_gaussian)) = 6

# Row names of the `:param` dimension when fitting a matrix. `gaussian_nobaseline` fits
# are stored with `y0 = 0`.
param_names(::typeof(gaussian)) = [:y0, :A, :mu, :sigma]
param_names(::typeof(gaussian_nobaseline)) = param_names(gaussian)
param_names(::typeof(double_gaussian)) = [:A_seeding, :mu_seeding, :sigma_seeding, :A_sase, :mu_sase, :sigma_sase]

params_struct(::typeof(gaussian), u) = GaussianParams(u)
params_struct(::typeof(gaussian_nobaseline), u) = GaussianParams(u)
params_struct(::typeof(double_gaussian), u) = DoubleGaussianParams(u)

CurveFit.ScalarModel(::typeof(gaussian)) = ScalarModel((p, x) -> gaussian(x, p[1], p[2], p[3], p[4]))
CurveFit.ScalarModel(::typeof(gaussian_nobaseline)) = ScalarModel((p, x) -> gaussian_nobaseline(x, p[1], p[2], p[3]))
CurveFit.ScalarModel(::typeof(double_gaussian)) = ScalarModel((p, x) -> double_gaussian(x, p[1], p[2], p[3], p[4], p[5], p[6]))

# Canonicalise fitted parameters: widths are only defined up to a sign, and for the
# double Gaussian the components can come out in either order.
function normalize_params!(::Union{typeof(gaussian), typeof(gaussian_nobaseline)}, u)
    u[end] = abs(u[end])
    u
end

function normalize_params!(::typeof(double_gaussian), u)
    u[3] = abs(u[3])
    u[6] = abs(u[6])

    if u[3] > u[6]
        for i in 1:3
            u[i], u[i + 3] = u[i + 3], u[i]
        end
    end

    u
end

Base.isnan(p::GaussianParams) = isnan(p.y0) || isnan(p.A) || isnan(p.mu) || isnan(p.sigma)
Base.isnan(p::DoubleGaussianParams) = isnan(p.seeding) || isnan(p.sase)

struct FitWorkspace{X, Y, S, U, C, F}
    safe_xdata::X
    safe_ydata::Y
    safe_sigma::S
    u0::U
    caches::Dict{Int, C}
    f::F

    function FitWorkspace(ydata; f=gaussian, xdata=1:length(ydata), u0=nothing, sigma=nothing, kwargs...)
        if isnothing(u0)
            u0 = zeros(eltype(ydata), parameter_count(f))
        end

        # Always floats, so the workspace can be reused with non-integer x data (e.g. a
        # DimVector lookup)
        safe_xdata = float.(collect(xdata))
        safe_ydata = collect(ydata)
        safe_sigma = isnothing(sigma) ? nothing : @view(sigma[1:end])

        cache = create_cache(f, @view(safe_xdata[1:end]), @view(safe_ydata[1:end]), safe_sigma)
        C = typeof(cache)
        caches = Dict{Int, C}(length(ydata) => cache)

        new{typeof(safe_xdata), typeof(safe_ydata), typeof(safe_sigma),
            typeof(u0), typeof(cache), typeof(f)}(safe_xdata, safe_ydata, safe_sigma, u0, caches, f)
    end

    FitWorkspace(ydata::AbstractDimVector; xdata=dim_xdata(ydata), kwargs...) = FitWorkspace(parent(ydata); xdata, kwargs...)
end

# Default x data for a DimVector: its lookup values, or 1:n if it has no lookup
function dim_xdata(ydata::AbstractDimVector)
    l = lookup(ydata, 1)
    if l isa DimensionalData.NoLookup
        1:length(ydata)
    else
        parent(l)
    end
end

function create_cache(f, xdata, ydata, sigma)
    u0 = zeros(eltype(ydata), parameter_count(f))
    nlfunc = NonlinearFunction{true}(ScalarModel(f); resid_prototype=similar(ydata))
    autodiff = AutoForwardDiff(; chunksize = NLS.pickchunksize(parameter_count(f)))
    polyalg = NLS.FastShortcutNLLSPolyalg(; autodiff)
    alg = NLS.NonlinearSolvePolyAlgorithm((polyalg.algs[2], polyalg.algs[3]))
    prob = NonlinearCurveFitProblem(nlfunc, u0, xdata, ydata, sigma)

    CurveFit.init(prob, alg)
end

function finite_mask!(workspace::FitWorkspace, xdata, ydata, sigma)
    i = 0
    for (x, y, s) in zip(xdata, ydata, something(sigma, eachindex(ydata)))
        if isfinite(x) && isfinite(y)
            i += 1
            workspace.safe_xdata[i] = x
            workspace.safe_ydata[i] = y

            if !isnothing(sigma)
                workspace.safe_sigma[i] = s
            end
        end
    end

    safe_xdata, safe_ydata = @views workspace.safe_xdata[1:i], workspace.safe_ydata[1:i]
    safe_sigma = isnothing(sigma) ? nothing : @view(workspace.safe_sigma[1:i])

    safe_xdata, safe_ydata, safe_sigma
end

function Base.show(io::IO, x::FitWorkspace)
    print(io, FitWorkspace, "()")
end

function guess_params!(::typeof(gaussian_nobaseline), u0, xdata, ydata)
    μ_idx = argmax(ydata)
    μ = xdata[μ_idx]
    A = max(1, ydata[μ_idx])
    xmin, xmax = extrema(xdata)
    σ = abs(xmax - xmin) / 4

    u0[1] = A
    u0[2] = μ
    u0[3] = σ
end

function guess_params!(::typeof(gaussian), u0, xdata, ydata)
    guess_params!(gaussian_nobaseline, @view(u0[2:end]), xdata, ydata)

    y0 = minimum(ydata)
    u0[1] = y0
end

function guess_params!(::typeof(double_gaussian), u0, xdata, ydata)
    T = eltype(u0)
    n = length(ydata)
    xmin, xmax = extrema(xdata)
    dx = T(abs(xmax - xmin) / max(1, n - 1))

    # SASE guess from the moments of the spectrum (negative noise clipped to 0). The
    # seeding peak is narrow so it only biases these slightly.
    wsum = zero(T)
    wxsum = zero(T)
    for (x, y) in zip(xdata, ydata)
        w = max(y, 0)
        wsum += w
        wxsum += w * x
    end

    if wsum > 0
        μ_sase = wxsum / wsum
        wvarsum = zero(T)
        for (x, y) in zip(xdata, ydata)
            wvarsum += max(y, 0) * (x - μ_sase)^2
        end
        σ_sase = max(sqrt(wvarsum / wsum), dx)
        # Amplitude from the area, assuming it all belongs to the SASE peak
        A_total = wsum * dx / (σ_sase * sqrt(T(2π)))
    else
        # Empty spectrum
        μ_sase = T((xmin + xmax) / 2)
        σ_sase = T(abs(xmax - xmin) / 4)
        A_total = one(T)
    end

    # Seeding guess from the largest excess over the SASE guess
    seeding_idx = firstindex(ydata)
    A_seeding = typemin(T)
    for i in eachindex(ydata)
        excess = ydata[i] - gaussian_nobaseline(xdata[i], A_total, μ_sase, σ_sase)
        if excess > A_seeding
            A_seeding = excess
            seeding_idx = i
        end
    end
    A_seeding = max(A_seeding, A_total / 10)
    μ_seeding = xdata[seeding_idx]

    # Seeding width from the half-maximum points of the excess
    half_max = A_seeding / 2
    left = seeding_idx
    while left > firstindex(ydata) &&
          ydata[left - 1] - gaussian_nobaseline(xdata[left - 1], A_total, μ_sase, σ_sase) > half_max
        left -= 1
    end
    right = seeding_idx
    while right < lastindex(ydata) &&
          ydata[right + 1] - gaussian_nobaseline(xdata[right + 1], A_total, μ_sase, σ_sase) > half_max
        right += 1
    end
    fwhm = abs(xdata[right] - xdata[left]) + dx
    σ_seeding = clamp(fwhm / (2 * sqrt(2 * log(T(2)))), dx / 2, σ_sase / 2)

    # Take the seeding peak's share of the area back out of the SASE amplitude
    A_sase = max(A_total - A_seeding * σ_seeding / σ_sase, A_total / 10)

    u0[1] = A_seeding
    u0[2] = μ_seeding
    u0[3] = σ_seeding
    u0[4] = A_sase
    u0[5] = μ_sase
    u0[6] = σ_sase
end

function fit_gaussian(ydata::AbstractVector; f=gaussian, xdata=1:length(ydata),
                      u0=nothing, workspace=nothing, sigma=nothing,
                      full_output_requested=false, params_as_array=true)
    if isnothing(workspace)
        workspace = FitWorkspace(ydata; f, xdata, u0, sigma)
    end
    if !isnothing(sigma) && isnothing(workspace.safe_sigma)
        throw(ArgumentError("Workspace is not configured for sigma, you must pass `sigma` when creating it: `FitWorkspace(ydata; sigma)`"))
    end

    safe_xdata, safe_ydata, safe_sigma = finite_mask!(workspace, xdata, ydata, sigma)

    n_params = parameter_count(workspace.f)
    if length(safe_ydata) < n_params
        nans = fill(eltype(ydata)(NaN), n_params)
        return params_as_array ? nans : params_struct(workspace.f, nans)
    end

    if isnothing(u0)
        guess_params!(workspace.f, workspace.u0, safe_xdata, safe_ydata)
    else
        workspace.u0 .= u0
    end

    cache = get!(workspace.caches, length(safe_ydata)) do
        create_cache(workspace.f, safe_xdata, safe_ydata, safe_sigma)
    end
    CurveFit.reinit!(cache; workspace.u0, x=safe_xdata, y=safe_ydata, sigma=safe_sigma)
    sol = solve!(cache)

    normalize_params!(workspace.f, sol.u)

    if full_output_requested
        return sol
    end

    u = NLS.SciMLBase.successful_retcode(sol) ? sol.u : fill(eltype(ydata)(NaN), n_params)
    params_as_array ? u : params_struct(workspace.f, u)
end

fit_gaussian(ydata::AbstractDimVector; xdata=dim_xdata(ydata), kwargs...) = fit_gaussian(parent(ydata); xdata, kwargs...)

function fit_gaussian!(output, ydatas::AbstractMatrix{T}; f=gaussian, scheduler=:greedy, ntasks=30) where {T}
    OMT.@tasks for (i, y) in enumerate(eachcol(ydatas))
        @set begin
            ntasks = ntasks
            scheduler = scheduler
        end
        @local workspace = FitWorkspace(first(eachcol(ydatas)); f)

        params = fit_gaussian(y; workspace)::Vector{T}
        if length(params) == size(output, 1)
            output[:, i] = params
        else
            # gaussian_nobaseline: store with y0 = 0
            output[2:end, i] = params
            output[1, i] = 0
        end
    end

    output
end

# TODO: use the lookup of the first dimension as the x data, like the DimVector method does
function fit_gaussian(ydatas::AbstractDimMatrix; f=gaussian, kwargs...)
    output = fill(eltype(ydatas)(NaN), (Dim{:param}(param_names(f)), dims(ydatas, 2)))
    fit_gaussian!(output, parent(ydatas); f, kwargs...)
end

function fit_gaussian(ydatas::AbstractMatrix; f=gaussian, kwargs...)
    output = fill(eltype(ydatas)(NaN), (Dim{:param}(param_names(f)), Dim{:dim_1}(size(ydatas, 2))))
    fit_gaussian!(output, ydatas; f, kwargs...)
end

function r2(data::AbstractVector, params::AbstractVector)
    ȳ = nanmean(data)
    y0, A, mu, sigma = params

    ss_res = 0.0
    ss_tot = 0.0
    for (xᵢ, yᵢ) in enumerate(data)
        if isfinite(yᵢ)
            ŷ = gaussian(xᵢ, y0, A, mu, sigma)
            ss_res += (yᵢ - ŷ)^2
            ss_tot += (yᵢ - ȳ)^2
        end
    end

    1 - ss_res / ss_tot
end

function r2(data::AbstractDimMatrix, fits::AbstractDimMatrix)
    new_dims = otherdims(data, :pixel)
    T = eltype(data)
    out = DimArray(fill(T(NaN), length.(new_dims)), new_dims)

    for (i, (ys, params)) in enumerate(zip(eachslice(data; dims=new_dims), eachslice(fits; dims=new_dims)))
        out[i] = r2(ys, params)
    end

    out
end

@kwdef struct SpikeWorkspace{Y, F}
    spike_params::Vector{GaussianParams{Float64}} = GaussianParams{Float64}[]
    peak_starts::Vector{Int} = Int[]
    peak_ends::Vector{Int} = Int[]
    sigma::Vector{Float32} = Float32[]
    smoothed_data::Y
    spike_data::Y
    fit_workspace::F
end

function SpikeWorkspace(data::AbstractVector; f=gaussian_nobaseline)
    fit_workspace = FitWorkspace(data; f)
    smoothed_data = similar(data)
    spike_data = similar(data)
    Y = typeof(smoothed_data)
    F = typeof(fit_workspace)
    SpikeWorkspace{Y, F}(; smoothed_data, spike_data, fit_workspace)
end

SpikeWorkspace(data::AbstractDimVector; kwargs...) = SpikeWorkspace(parent(data); kwargs...)

Base.show(io::IO, workspace::SpikeWorkspace) = print(io, SpikeWorkspace, "()")

function find_spikes(data::AbstractVector, sel=1:length(data); workspace=nothing, w=20, smoothing_window=5, min_prominence=100)
    if isnothing(workspace)
        workspace = SpikeWorkspace(data)
    end

    spike_params = workspace.spike_params
    empty!(spike_params)
    all_peaks = nothing

    # Select a region to look for peaks within
    x_min, x_max = extrema(sel)
    data = movmean!(workspace.smoothed_data, data, smoothing_window)
    clipped_data = @view data[x_min:x_max]
    if isempty(data)
        return spike_params, nothing
    end

    # Find the peaks
    peaks = Peaks.findpeaks(data, w; strict=false, proms=(; min=min_prominence, max=Inf))

    peak_starts = workspace.peak_starts
    peak_ends = workspace.peak_ends
    empty!(peak_starts)
    empty!(peak_ends)
    for (i, (x, w)) in enumerate(zip(peaks.indices, peaks.widths))
        # Sometimes Peaks.jl may give nan widths
        if !(isfinite(x) && isfinite(w))
            continue
        end

        push!(peak_starts, max(1, round(Int, x - w)))
        push!(peak_ends, min(length(clipped_data), round(Int, x + w)))
    end

    # Fit the peaks
    spike_data = workspace.spike_data
    resize!(spike_data, length(sel))
    sorted_peaks = sortperm(peaks.proms)[1:min(10, end)]
    for (i, peak_idx) in enumerate(sorted_peaks)
        fill!(spike_data, 0)

        spike_start = peak_starts[peak_idx]
        spike_end = peak_ends[peak_idx]
        spike_range = spike_start:spike_end
        spike_data[spike_range] .= clipped_data[spike_range]

        approx_width = length(spike_range)
        fit_start = max(1, spike_start - approx_width)
        fit_end = min(length(spike_data), spike_end + approx_width)

        x = fit_gaussian(@view spike_data[fit_start:fit_end];
                         workspace=workspace.fit_workspace,
                         params_as_array=false)::GaussianParams

        if !isnan(x)
            shifted_mu = x.mu + fit_start
            push!(spike_params, GaussianParams(x.y0, x.A, shifted_mu, x.sigma))
        end
    end

    spike_params
end

find_spikes(data::AbstractDimVector, sel; kwargs...) = find_spikes(parent(data), sel; kwargs...)

function find_spikes(data::AbstractDimMatrix, sel=axes(data, 1); w=2, smoothing_window=5, min_prominence=18)
    dim = first(dims(data))
    other_dims = otherdims(data, dim)
    spike_fits = DimArray(fill(NaN32, (4, 10, length.(other_dims)...)),
                          (Dim{:param}([:y0, :A, :sigma, :mu]), Dim{:spike}(), other_dims...))
    spikes = DimArray(fill(NaN32, (5, 10, length.(other_dims)...)),
        (Dim{:property}([:height, :center, :start, :end, :width]), Dim{:spike}(), other_dims...))

    OMT.@tasks for i in CartesianIndices(other_dims)
        @set begin
            ntasks = 40
            scheduler = :greedy
        end
        @local workspace = SpikeWorkspace(data[:, 1])

        params = find_spikes(parent(@view(data[:, i])), sel; w, smoothing_window, min_prominence, workspace)
        for spike_i in eachindex(params)
            p = params[spike_i]
            spike_fits[:, spike_i, i] .= (p.y0, p.A, p.mu, p.sigma)
            # spikes[:, spike_i, i] .= _spikes[peak=spike_i]
        end
    end

    spike_fits, spikes
end

@compile_workload begin
    for T in (Float32, Float64)
        true_params = [1, 20, 25, 10]
        tids = UInt.(collect(1:2))
        xs = 1:50
        data = DimArray(zeros(T, length(xs), length(tids)),
                        (; pixel=DimensionalData.NoLookup(), trainId=tids))

        for spectrum in eachslice(data; dims=:trainId)
            @. spectrum = gaussian(xs, true_params[1], true_params[2], true_params[3], true_params[4])
        end

        fits = fit_gaussian(data)
        r2(data, fits)
    end
end

end # module HirexFitting

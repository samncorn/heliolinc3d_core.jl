module heliolinc_core

using StaticArrays
using LinearAlgebra
using Rotations
using NearestNeighbors
using Roots
using dbscan
using Kepler
# using Healpix
# using Random
# using DifferentialEquations
# using Enzyme
# using ForwardDiff
# using DiffResults
# using Statistics
# using Distributions
using StatsBase
using Dates
using LoggingExtras
using ProgressMeter
using Printf
using Dates

const I3 = SMatrix{3, 3}(diagm(ones(3)))
const I6 = SMatrix{6, 6}(diagm(ones(6)))

struct MaximumIterationError
    i::Int
end


""" ASSUMES ANGLES TO BE IN RADIANS

pos and vel are of the observer at time of measurement
"""
struct AnglesMeas{T}
    time :: T
    # ang  :: SVector{3, T}
    # dang :: SVector{3, T}
    ang  :: SVector{3, T}
    dang :: SVector{3, T}
    pos  :: SVector{3, T}
    vel  :: SVector{3, T}
end

AnglesMeas(time, ang, pos) = AnglesMeas(time, ang, zero(ang), pos, zero(pos))

struct Hypothesis{T}
    r0   :: T
    dr0  :: T
    ddr0 :: T
    t0   :: T
    gm   :: T
    c    :: T
end

struct ClusteringParams{T}
    cluster_radius :: T
    velocity_scale :: T
    n_threads      :: Int
    max_rms        :: T
    min_pts        :: Int
    max_pts        :: Int
    # healpix_order  :: Int
end

struct OrbitBounds{T}
    r_min   :: T
    e_min   :: T
    e_max   :: T
    rho_min :: T
    i_min   :: T
    i_max   :: T
end

const default_orbit_bounds = OrbitBounds(0.0, 0.0, 1.0, 0.0, 0.0, 90.0)

struct DetectionCluster{T}
    ids :: Vector{Int}
    pos :: SVector{3, T}
    vel :: SVector{3, T}
end

# struct ArrowPartition{T}
#     points :: Vector{T}
#     index  :: Vector{Int}
# end

include("math.jl")
include("tracklets.jl")
include("arrows.jl")
include("partitions.jl")
include("clustering.jl")
# include("least_squares.jl")
# include("herget.jl")
# include("differential_correction.jl")
include("linkages.jl")

""" 
angles as cartesian unit vectors. 

"""
function heliolinc(tracklets, angles, times, obs, hypothesis, clustering_params, orbit_bounds; n_threads = 1, R2ECL = I, angle_iters = 0, progress_bar = false)
    # compute arrows
    @info "  computing arrows..."
    t0 = now()
    arrows, trkids, branch = make_arrows(tracklets, angles, times, obs, hypothesis, orbit_bounds; 
        n_threads = n_threads, max_iters = angle_iters, R2ECL = R2ECL, min_range = orbit_bounds.rho_min,
        )
    tf = now()
    @info @sprintf("  found %i (%.3e bytes) valid tracklets in %s", length(arrows), sizeof(arrows), canonicalize(tf-t0))

    # scale the velocities
    Threads.@threads for i in eachindex(arrows)
        pos = SVector{3}(view(arrows[i], 1:3))
        vel = SVector{3}(view(arrows[i], 4:6)) .* clustering_params.velocity_scale
        arrows[i] = SA[pos..., vel...]
    end

    @info "  running DBSCAN..."
    t0 = now()
    # labels = DBSCAN(
    #     partitions, 
    #     res, 
    #     clustering_params.cluster_radius, 
    #     clustering_params.min_pts; 
    #     n_threads = n_threads, 
    #     metric = metric
    #     )
    # labels = dbscan.DBSCAN_cells(arrows, clustering_params.cluster_radius, clustering_params.min_pts; n_threads = n_threads, max_pts = clustering_params.max_pts,)
    @info "  collecting clusters..."
    # clusters = cluster_all(
    #     arrows, 
    #     clustering_params.cluster_radius, 
    #     clustering_params.min_pts;
    #     n_threads = n_threads, 
    #     max_pts = clustering_params.max_pts
    # )

    # # cluster_states = [
    # #     mean(arrows[ids] for ids in clusters) ./ SVector{6}(1.0, 1.0, 1.0, clustering_params.velocity_scale, clustering_params.velocity_scale, clustering_params.velocity_scale)
    # # ]

    # cluster_states = Vector{SVector{6, Float64}}(undef, 0)
    # cluster_sigmas = Float64[]
    # linkages       = Vector{Vector{Int}}(undef, 0)
    # for ids in clusters
    #     states = arrows[ids]
    #     # link   = unique(Iterators.flatten(tracklets[trkids[ids]]))
    #     link   = unique(Iterators.flatten(Iterators.map(i -> tracklets[trkids[i]], ids)))

    #     push!(linkages, link)
    #     push!(cluster_states, mean(states) ./ SVector{6}(1.0, 1.0, 1.0, clustering_params.velocity_scale, clustering_params.velocity_scale, clustering_params.velocity_scale))
    #     push!(cluster_sigmas, norm(std(states)))
    # end

    # return linkages, cluster_states, cluster_sigmas

    labels = cluster_unique(
        arrows, 
        clustering_params.cluster_radius, 
        clustering_params.min_pts;
        n_threads = n_threads, 
        max_pts = clustering_params.max_pts
    )
    tf = now()
    @info @sprintf("  collected clusters in %s", canonicalize(tf-t0))
    tf = now()
    @info "  clustered arrows in $(canonicalize(tf-t0))"

    # unscale the velocities
    Threads.@threads for i in eachindex(arrows)
        pos = SVector{3}(view(arrows[i], 1:3))
        vel = SVector{3}(view(arrows[i], 4:6)) ./ clustering_params.velocity_scale
        arrows[i] = SA[pos..., vel...]
    end

    # # filter out noise points
    mask = labels .!= 0
    keepat!(labels, mask)
    keepat!(arrows, mask)
    keepat!(trkids, mask)
    keepat!(branch, mask)
    return labels, arrows, trkids, branch
end

""" cell search to return neighborhood of each point
"""
function heliolinc_cells(tracklets, angles, times, obs, hypothesis, clustering_params, orbit_bounds; n_threads = 1, R2ECL = I, angle_iters = 0, progress_bar = false)
    @info "  computing arrows..."
    t0 = now()
    arrows, trkids, branch = make_arrows(tracklets, angles, times, obs, hypothesis, orbit_bounds; 
        n_threads = n_threads, max_iters = angle_iters, R2ECL = R2ECL, min_range = orbit_bounds.rho_min,
        )
    tf = now()
    @info @sprintf("  found %i (%.3e bytes) valid tracklets in %s", length(arrows), sizeof(arrows), canonicalize(tf-t0))

    # scale the velocities
    Threads.@threads for i in eachindex(arrows)
        pos = SVector{3}(view(arrows[i], 1:3))
        vel = SVector{3}(view(arrows[i], 4:6)) .* clustering_params.velocity_scale
        arrows[i] = SA[pos..., vel...]
    end

    @info "  collecting clusters..."
    t0 = now()
    clusters = cluster_cells(
        arrows, 
        clustering_params.cluster_radius, 
        clustering_params.min_pts; 
        n_threads = n_threads, 
        max_pts = clustering_params.max_pts
    )
    tf = now()
    @info @sprintf("  collected clusters in %s", canonicalize(tf-t0))

    return clusters, trkids, branch
end

""" kd-tree search to return neighborhood of each point
"""
function heliolinc_tree(tracklets, angles, times, obs, hypothesis, clustering_params, orbit_bounds; n_threads = 1, R2ECL = I, angle_iters = 0, progress_bar = false)
    @info "  computing arrows..."
    t0 = now()
    arrows, trkids, branch = make_arrows(tracklets, angles, times, obs, hypothesis, orbit_bounds; 
        n_threads = n_threads, max_iters = angle_iters, R2ECL = R2ECL, min_range = orbit_bounds.rho_min,
        )
    tf = now()
    @info @sprintf("  found %i (%.3e bytes) valid tracklets in %s", length(arrows), sizeof(arrows), canonicalize(tf-t0))

    # scale the velocities
    Threads.@threads for i in eachindex(arrows)
        pos = SVector{3}(view(arrows[i], 1:3))
        vel = SVector{3}(view(arrows[i], 4:6)) .* clustering_params.velocity_scale
        arrows[i] = SA[pos..., vel...]
    end

    @info "  building tree..."
    t0 = now()
    tree = KDTree(arrows)
    tf = now()
    @info @sprintf("  built tree in %s", canonicalize(tf-t0))

    @info "  collecting clusters..."
    t0 = now()
    clusters = inrange(tree, arrows, clustering_params.cluster_radius)
    tf = now()
    @info @sprintf("  collected clusters in %s", canonicalize(tf-t0))

    @info "  filtering clusters..."
    t0 = now()
    filter!(x -> clustering_params.min_pts <= length(x) <= clustering_params.max_pts, clusters)
    tf = now()
    @info @sprintf("  filtered clusters in  in %s", canonicalize(tf-t0))

    return clusters, trkids, branch
end

# function heliolinc_tree(tracklets, angles, times, obs, hypothesis, clustering_params, orbit_bounds; n_threads = 1, R2ECL = I, angle_iters = 0, progress_bar = false)
#     @info "  computing arrows..."
#     t0 = now()
#     arrows, trkids = make_arrows(tracklets, angles, times, obs, hypothesis, orbit_bounds; n_threads = n_threads, max_iters = angle_iters, R2ECL = R2ECL)
#     tf = now()
#     @info @sprintf("  found %i (%.3e bytes) valid tracklets in %s", length(arrows), sizeof(arrows), canonicalize(tf-t0))

#     # scale the velocities
#     Threads.@threads for i in eachindex(arrows)
#         pos = SVector{3}(view(arrows[i], 1:3))
#         vel = SVector{3}(view(arrows[i], 4:6)) .* clustering_params.velocity_scale
#         arrows[i] = SA[pos..., vel...]
#     end
#     cluster_ids, cluster_trks = find_clusters(arrows, clustering_params.cluster_radius, clustering_params.min_pts, clustering_params.max_pts; progress_bar = progress_bar)

#     # unscale the velocities
#     Threads.@threads for i in eachindex(arrows)
#         pos = SVector{3}(view(arrows[i], 1:3))
#         vel = SVector{3}(view(arrows[i], 4:6)) ./ clustering_params.velocity_scale
#         arrows[i] = SA[pos..., vel...]
#     end

#     return cluster_ids, cluster_trks, arrows, trkids
# end

# """ Use a kd-tree to generate a cluster for each valid seed point
# """
# function find_clusters_tree(points, radius, min_pts, max_pts; progress_bar = false)
#     prog = Progress(length(points), desc = "clustering...", enabled = progress_bar)
#     # find the clusters
#     cluster_ids  = Int[]
#     cluster_trks = Int[]
#     tree = KDTree(points)
#     neighbors = Int[]
#     for (i, p) in pairs(points)
#         inrange!(neighbors, tree, p, radius)
#         if min_pts <= length(neighbors) <= max_pts
#             append!(cluster_ids, [i for _ in neighbors])
#             append!(cluster_trks, neighbors)
#         end

#         next!(prog)
#         empty!(neighbors)
#     end
#     finish!(prog)
#     return cluster_ids, cluster_trks
# end

# """ sort points into appropriately sized bins (~1/2 the cluster radius). If a cell has enough points, generate a cluster
# with it and the surrounding cells.

# Will result in some duplicate clusters, but not as many as with find_clusters_tree. 
# """
# function find_clusters_cells(points::AbstractVector{SVector{D, T}}, radius, min_pts, max_pts; progress_bar = false) where {D, T}
#     prog = Progress(length(points), desc = "clustering...", enabled = progress_bar)
#     # initialize return arrays
#     cluster_ids  = Int[]
#     cluster_trks = Int[]

#     # map points to cells
#     cells = Dict{SVector{D, Int32}, Vector{Int}}()
#     center = mean(points)

#     # use the half radius, so so that cells can be combined into appropriately sized clusters
#     prog1 = Progress(length(points), enabled = progress_bar)
#     for (i, p) in pairs(points)
#         cell = SVector{D, Int32}(dbscan.find_cell(p - center, radius/2))
#         if haskey(cells, cell)
#             push!(cells[cell], i)
#         else
#             cells[cell] = [i]
#         end
#         next!(prog1)
#     end
#     finish!(prog1)

#     # loop through cells, generating clusters, adding to return arrays


#     return cluster_ids, cluster_trks
# end

# """deprecated"""
# function _heliolinc(detections, tracklets, hypothesis, clustering_params, orbit_bounds; n_threads = 1)
#     weights = SVector{6}(
#         1.0, 
#         1.0, 
#         1.0, 
#         clustering_params.velocity_scale, 
#         clustering_params.velocity_scale, 
#         clustering_params.velocity_scale,
#         ) .^ 2
#     metric = WeightedEuclidean(weights)

#     # compute arrows
#     @info "  computing arrows..."
#     t0 = now()
#     arrows, trkids = make_arrows(detections, tracklets, hypothesis, orbit_bounds; n_threads = n_threads)
#     tf = now()
#     @info "  found $(length(arrows)) valid tracklets in $(canonicalize(tf-t0))"
#     # @info ""

#     # sort arrows into healpix partitions
#     # chunk map maps indices in the total array to healpixels
#     @info "  sorting arrows into chunks..."
#     res = Resolution(2^clustering_params.healpix_order)
#     t0 = now()
#     partitions, chunk_map = hp_partition_arrows(arrows, res; n_threads = n_threads)
#     tf = now()
#     @info "partitioned arrows in $(canonicalize(tf-t0))"

#     @info "  running DBSCAN..."
#     t0 = now()
#     labels = DBSCAN(
#         partitions, 
#         res, 
#         clustering_params.cluster_radius, 
#         clustering_params.min_pts; 
#         n_threads = n_threads, 
#         metric = metric
#         )
#     tf = now()
#     @info "  clustered arrows in $(canonicalize(tf-t0))"

#     # @info " collecting cluster labels..."
#     # # collect 
#     # t0 = now()
#     # clusters = collect_labels(labels)
#     # tf = now()
#     # @info "  collected clusters in $(canonicalize(tf-t0))"

#     # filter out noise points
#     mask = labels .!= 0

#     return labels[mask], arrows[mask], trkids[mask]
# end

# struct HLCluster{T1, T2}
#     arrows::Vector{T1}
#     trkids::Vector{T2}
# end

# function collect_clusters(labels::AbstractVector{T1}, arrows::AbstractVector{T2}, trkids::AbstractVector{T3}) where {T1, T2, T3}
#     clusters = Dict{T1, HLCluster{T2, T3}}()
#     clusters    = Dict{Int, Vector{T}}()
#     for (i, l) in pairs(labels)
#         if haskey(clusters, l)
#             clusters[l] = HLCluster([arrows[i]], [trkids[i]])
#         else
#             # push!(clusters[l], i)
#             push!(clusters[l].arrows, arrows[i])
#             push!(clusters[l].trkids, trkids[i])
#         end
#     end
#     return clusters
# end

end # module

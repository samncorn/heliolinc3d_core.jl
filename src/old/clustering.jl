

function cluster_detections(
    detections, tracklets,
    r0, dr0, t0, gm,
    cluster_radius;
    scaling = 1.0, c = Inf,
    state_estimate   = CentralDifference1, 
    state_clustering = KeplerianPropagation,
    distance_model   = QuadraticDistance,
    r_min = 0.0,
    e_min = 0.0, e_max = 1.0, 
    n_threads = 1, max_iter = 10, ddr0_tol = 1e-6,
    max_rms = Inf, max_pts = 1000,
    min_pts = 3, rho_min = 0.0,
    )

    # instantiate healpix partitioning

    # configure arrow iterator?

    t_start = now()
    arrows, valid_trks, times = make_arrows(
        detections, 
        tracklets, 
        r0, dr0, t0, gm; 
        c = c, 
        state_estimate = state_estimate, 
        state_clustering = state_clustering,
        distance_model = distance_model,
        r_min = r_min,
        e_min = e_min, e_max = e_max,
        rho_min = rho_min,
        n_threads = n_threads, max_iter = max_iter
    )

    if length(arrows) < 3 # minimum number of points to form a cluster
        return nothing
    end

    prop_arrows = Vector{SVector{6, Float64}}(undef, length(arrows))
    for (i, arr) in enumerate(arrow)
        pos = SVector{3}(@view arr[1:3])
        vel = SVector{3}(@view arr[4:6])
        pos, vel = Kepler.solve(pos, vel, t0 - times[i], gm)
        prop_arrows[i] = SVector{6}((pos..., vel...))
    end

    t_stop = now()
    # @info "found $(length(arrows)) in $(canonicalize(t_stop - t_start))"

    weights  = SVector{6}(1.0, 1.0, 1.0, scaling, scaling, scaling) .^ 2
    metric   = WeightedEuclidean(weights)

    t_start = now()
    clusters = dbscan.DBSCAN(prop_arrows, cluster_radius, min_pts; n_threads = n_threads, metric = metric, max_pts = max_pts)
    t_stop = now()
    # @info "found $(length(clusters)) in $(canonicalize(t_stop - t_start))"

    if length(clusters) == 0
        return nothing
    end

    # 3-sigma clipping
    # clusters = collect(Iterators.map(c -> clip_cluster!(c, prop_arrows), clusters))
    # for c in clusters
    #     clip_cluster!(c, prop_arrows)
    # end
    # drop wide dispersion clusters
    clusters = collect(clusters)
    # clusters = filter!(c -> check_cluster(view(prop_arrows, c), max_rms), clusters)
    # if length(clusters) == 0
    #     return nothing
    # end
    labels   = get_labels(clusters)

    idx    = reduce(vcat, clusters)
    arrows = arrows[idx]
    times  = times[idx]
    dets   = tracklets[valid_trks[idx]]
    # collect clusters and labels into arroy to return
    return dets, arrows, times, labels
end

function clip_cluster_point(arrow, mean_arrow, sigma; n_sigma = 3)
    return sum(((arrow - mean_arrow) .^ 2) ./ (sigma .^ 2)) < n_sigma
end

function clip_cluster!(c, arrows; n_sigma = 3)
    c_arrows = view(arrows, c)
    mean_state = mean(c_arrows)
    sigma      = std(c_arrows)
    filter!(i -> clip_cluster_point(arrows[i], mean_state, sigma; n_sigma = n_sigma), c)
end

# function check_cluster(arrows, max_rms)
#     sigma = std(norm.(arrows .- mean(arrows)))
#     # arr_eig = eigvals(cov(arrows))
#     # sigma   = sqrt(maximum(arr_eig))
#     return sigma < max_rms
# end

function count_dets(trks)
    return length(collect(unique(Iterators.flatten(trks))))
end

function get_labels(clusters)
    n_arrs = reduce(+, Iterators.map(length, clusters))
    labels = zeros(Int, n_arrs)
    j = 1
    for (i, c) in enumerate(clusters)
        for _ in enumerate(c)
            labels[j] = i
            j += 1
        end
    end
    return labels
end
"""
v_max, arc_min USING RADIANS
"""
function find_tracklets(angles, times, v_min, v_max, dt_min, dt_max; arc_min = 0.0, progress = false, n_threads = 1)
    n = length(angles)
    @assert n == length(times)
    prog = Progress(n; enabled = progress)

    # angles = map(((a, b),) -> SVector{})
    tree      = KDTree(angles)
    r_search  = v_max * dt_max
    tracklets = Vector{Any}(undef, n_threads)

    Threads.@threads for i_t in 1:n_threads
        tracklets[i_t] = Tuple{Int, Int}[]
        neighbors = Int[]
        for i in i_t:n_threads:n
            p_i = angles[i]

            empty!(neighbors)
            inrange!(neighbors, tree, p_i, r_search)
            for j in neighbors
                i == j && continue

                dt_ij = times[j] - times[i]
                !(dt_min < dt_ij < dt_max) && continue

                dot_ij = dot(angles[i], angles[j])
                rr_ij  = norm(angles[i])*norm(angles[j])
                ang = dot_ij / rr_ij <= 1.0 ? acos(dot_ij / rr_ij) : 0.0 # should handle edge cases where > 1
                # ang = haversine(di.ang[1], dj.ang[1], di.ang[2], dj.ang[2])
                v_ij = ang/dt_ij
                if v_min < v_ij < v_max
                    push!(tracklets[i_t], (i, j))
                end
            end
            next!(prog)
        end
    end
    finish!(prog)
    return reduce(vcat, tracklets)
end

# function find_tracklets_for_

"""
Exclude some detections from viable tracklets

v_max, arc_min USING RADIANS
"""
function find_tracklets(angles, times, v_min, v_max, dt_min, dt_max, exclude; arc_min = 0.0, progress = false, n_threads = 1)
    n = length(angles)
    @assert n == length(times)
    prog = Progress(n; enabled = progress)

    # angles = map(((a, b),) -> SVector{})
    tree = KDTree(angles)
    r_search = v_max * dt_max
    tracklets = Vector{Any}(undef, n_threads)
    Threads.@threads for i_t in 1:n_threads
        tracklets[i_t] = Tuple{Int, Int}[]
        neighbors = Int[]
        for i in i_t:n_threads:n
            if exclude[i] 
                continue
            end
            p_i = angles[i]

            empty!(neighbors)
            inrange!(neighbors, tree, p_i, r_search)
            for j in neighbors
                # exclude[j] && continue
                # i == j && continue
                if exclude[j] || i == j
                    continue
                end

                dt_ij = times[j] - times[i]
                !(dt_min < dt_ij < dt_max) && continue

                dot_ij = dot(angles[i], angles[j])
                rr_ij  = norm(angles[i])*norm(angles[j])
                ang = dot_ij / rr_ij <= 1.0 ? acos(dot_ij / rr_ij) : 0.0 # should handle edge cases where > 1
                # ang = haversine(di.ang[1], dj.ang[1], di.ang[2], dj.ang[2])
                v_ij = ang/dt_ij
                if v_min < v_ij < v_max
                    push!(tracklets[i_t], (i, j))
                end
            end
            next!(prog)
        end
    end
    finish!(prog)
    return reduce(vcat, tracklets)
end

function find_tracklets(detections, v_max, dt_min, dt_max; n_threads = 1, arc_min = 0.0, progress = false)
    arc_min_rad = deg2rad(arc_min)
    N = length(detections)
    prog = Progress(N, enabled = progress)
    xyz = [SVector(d.ang) for d in detections]
    # xyz = map(d -> SVector{3}(sphere2cart(d.ang[1], d.ang[2])), detections)
    tree = KDTree(xyz)

    r_search = v_max * dt_max # TODO get the chord length
    tracklets = Vector{Any}(undef, n_threads)
    Threads.@threads for i_t in 1:n_threads
        _tracklets = Tuple{Int, Int}[]
        neighbors = Int[]
        for i in i_t:n_threads:N
            p_i = xyz[i]
            empty!(neighbors)
            inrange!(neighbors, tree, p_i, r_search)
            for j in neighbors
                i == j && continue

                di = detections[i]
                dj = detections[j]

                dt_ij = dj.time - di.time
                !(dt_min < dt_ij < dt_max) && continue

                dot_ij = dot(xyz[i], xyz[j])
                rr_ij  = norm(xyz[i])*norm(xyz[j])
                ang = dot_ij / rr_ij <= 1.0 ? acos(dot_ij / rr_ij) : 0.0 # should handle edge cases where > 1
                # ang = haversine(di.ang[1], dj.ang[1], di.ang[2], dj.ang[2])
                if arc_min_rad < ang <= v_max*dt_ij
                    push!(_tracklets, (i, j))
                end
            end
            next!(prog)
        end
        tracklets[i_t] = _tracklets
    end
    finish!(prog)
    return reduce(vcat, tracklets)
end

function count_tracklets(detections, v_max, dt_min, dt_max; n_threads = 1, arc_min = 0.0, progress = false)
    arc_min_rad = deg2rad(arc_min)
    N = length(detections)
    prog = Progress(N, enabled = progress)
    xyz = [d.ang for d in detections]
    tree = KDTree(xyz)

    r_search = v_max * dt_max # TODO get the chord length
    tracklets = Vector{Int}(undef, n_threads)
    Threads.@threads for i_t in 1:n_threads
        n = 0
        neighbors = Int[]
        for i in i_t:n_threads:N
            p_i = xyz[i]
            empty!(neighbors)
            inrange!(neighbors, tree, p_i, r_search)
            for j in neighbors
                i == j && continue

                di = detections[i]
                dj = detections[j]

                dt_ij = dj.time - di.time
                !(dt_min < dt_ij < dt_max) && continue

                # dot_ij = dot(xyz[i], xyz[j])
                # rr_ij  = norm(xyz[i])*norm(xyz[j])
                ang = haversine(di.ang[1], dj.ang[1], di.ang[2], dj.ang[2])
                if arc_min_rad < ang <= v_max*dt_ij
                    n += 1
                end
            end
            next!(prog)
        end
        tracklets[i_t] = n
    end
    finish!(prog)
    return reduce(+, tracklets)
end

# function _loop_tracklets(f, tree, ang, idx, dt_min, dt_max, v_max)
#     r_search = v_max * dt_max
#     neighbors = UInt[]
#     for (i, p1) in pairs(ang)
#         inrange!(neighbors, tree, p_i, r_search)
#         for j in neighbors
#             idxs[i] == j && continue

#         end
#         empty!(neighbors)
#     end
# end
# struct Track{T}
#     orbit::Kepler.Cartesian{T}
#     score::Float64{T} # intended to be the redcued chi^2, but doesn't have to be, just should be consistent
#     ids::Vector{Int64}
# end

# struct TrackTable{T}
#     ra::Vector{T}
#     dec::Vector{T}
#     time::Vector{T}
#     obsPos::Vector{SVector{T}}
#     labels::Vector{Int64}
#     Tracks::Vector{Linkage{T}}
# end

# TrackTable(ra, dec, time, obsPos) = length(ra) == length(dec) == length(time) == length(obsPos) ? TrackTable(ra, dec, time, obsPos, zeros(Int64, length(ra)), []) : throw("inconstistent measurement vectors provided")

""" Try to insert a track to a track table, maintaining the one-to-one relation of observation to track
"""
# function push!(tbl::TrackTable{T}, orbit::Kepler.Cartesian{T}, ids::Vector{Int64}) where {T}
#     if check_conflict(tbl, ids)
#         # try to merge 
#     else
#         # we can insert safely
#         track = fit_track(tbl, orbit, ids)
#         push!(tbl.linakges, track)
#     end
# end

# function check_conflict(tbl::TrackTable{T}, ids::Vector{Int64}) where {T}
#     for d in ids
#         if d > length(tbl.labels)
#             throw("invalid id for insertion to table")
#         else
#             if tbl.labels[d] != 0
#                 return true
#             end
#         end
#     end
#     return false
# end

# function fit_track(tbl::TrackTable{T}, orbit::Kepler.Cartesian{T}, ids::Vector{Int64}; min_obs = 6)
#     ra   = tbl.ra[ids]
#     dec  = tbl.dec[ids]
#     time = tbl.time[ids]
#     obs  = tbl.obsPos[ids]

#     if length(ids) > min_obs
#         # RANSAC to check for outliers
#         orbit, score, mask = nothing # RANSAC()
#         return Track(orbit, score, ids[mask])
#     else
#         # not enough observations to reject outliers, 
#         orbit, score = Kepler.fit_orbit(orbit, ra, dec, time, obs; c = Inf)
#         return Track(orbit, score, ids)
#     end
# end


""" Convolve a tracklet window with the detection times

times needs to be sorted
"""
function tracklet_coverage(times; dt_min = 5/60/24, dt_max = 2/24, window = 0.5)
    if length(times) <= 1
        return false
    end
    n = length(times)

    tracklets = [false for _ in 1:n]
    for (i, j) in Iterators.product(1:n, 1:n)
        if i != j && dt_min <= times[j] - times[i] <= dt_max
            tracklets[i] = true
        end
    end

    # tracklets = [dt_min <= times[i+1] - times[i] <= dt_max for i in 1:n-1]

    coverage = 0.0
    for i in 1:n-1
        if tracklets[i]
            for j in (i+1):n
                if times[j] - times[i] < window
                    tracklets[j] = false
                end
            end

            coverage += window
        end
    end
    return coverage
end

function is_discoverable(times; window = 0.5, n_nights = 3, dt_min = 5/60/24, dt_max = 2/24)
    coverage = tracklet_coverage(sort(times); window = window, dt_min = dt_min, dt_max = dt_max)
    return coverage >= n_nights*window
end

function available_tracklets(times; dt_min = 0.0, dt_max = 2/24)
    tracklets = Tuple{Int, Int}[]
    for ((i1, t1), (i2, t2)) in Iterators.product(enumerate(times), enumerate(times))
        dt = t2 - t1
        if dt_min < dt < dt_max
            push!(tracklets, (i1, i2))
        end
    end
    return tracklets
end

function is_discoverable(times, ra, dec; window = 0.5, n_nights = 3, dt_min = 5/60/24, dt_max = 2/24, v_min = 0.0, v_max = 10.0, deg = true)
    # more precise check on angular velocities
    # find potential tracklets. Drop ones outside velocity bounds. Convolve remianing for time span coverage
    tracklets  = available_tracklets(times; dt_min = dt_min, dt_max = dt_max)
    trk_speeds = [haversine(ra[i], ra[j], dec[i], dec[j]; deg = deg)/abs(times[i] - times[j]) for (i, j) in tracklets]
    mask       = v_min .< trk_speeds .< v_max
    _times     = times[reduce(vcat, Iterators.flatten(tracklets[mask]); init = Int[])]
    coverage   = tracklet_coverage(sort(_times); window = window, dt_min = dt_min, dt_max = dt_max)
    return coverage >= n_nights*window
end

function dedupe_times(times; dt = 1e-5)
    out = [Int[]]
    has_conflict = Int[]
    # find initial set with no conflicts, record what cannot be inserted
    for i in eachindex(times)
        can_add = true
        for j in out[1]
            if abs(times[i] - times[j]) < dt
                can_add = false
                push!(has_conflict, i)
            end
        end
        if can_add
            push!(out[1], i)
        end
    end

    for j in has_conflict
        push!(out, [j])
    end

    for i in eachindex(times)
        for chunk in out
            can_add = true
            for j in chunk
                if abs(times[i] - times[j]) < dt
                    can_add = false
                end
            end
            if can_add
                push!(chunk, i)
            end
        end
    end

    return out
end
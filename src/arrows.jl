# @enum StateEstimate CentralDifference1 CentralDifference2 KeplerianIterate
# @enum StateClustering KeplerianPropagation MomentumEccentricity
# @enum DistanceModel LinearDistance QuadraticDistance AnalyticModel

""" better and multithreaded
"""
function make_arrows(detections, tracklets, hypothesis, orbit_bounds; n_threads = 1, project_tracklet::Function = project_tracklet_quadratic)
    n_tracklets = length(tracklets)
    n_possible  = 2n_tracklets 
    arrows = Vector{SVector{6, Float64}}(undef, n_possible)
    trkids = Vector{Int}(undef, n_possible)
    mask   = BitArray(undef, n_possible) 
    mask  .= 0 # just in case
    
    Threads.@threads for i_t in 1:n_threads
        for i_trk in i_t:n_threads:n_tracklets
            (i, j) = tracklets[i_trk]
            for branch in (false, true)
                pos, vel, t = project_tracklet(detections[i], detections[j], hypothesis; branch = branch)
                rho12       = norm(pos - (detections[j].pos - detections[i].pos) / (detections[j].time - detections[i].time) * (t - detections[i].time))
                if rho12 >= orbit_bounds.rho_min && isin_bounds(pos, vel, orbit_bounds, hypothesis.gm)
                    pos, vel = Kepler.solve(pos, vel, hypothesis.t0 - t, hypothesis.gm)
                    i_arr = 2i_trk - branch
                    arrows[i_arr] = SA[pos..., vel...]
                    trkids[i_arr] = i_trk
                    mask[i_arr]   = true
                end
            end
        end
    end
    return arrows[mask], trkids[mask]
end

"not yet multithreaded"
function make_arrows_hp!(healpix_partitions, res, detections, tracklets, hypothesis, orbit_bounds; n_threads = 1, project_tracklet::Function = project_tracklet_quadratic)
    @assert length(healpix_partitions) == res.numOfPixels
    for i_trk in eachindex(tracklets) # multithread here, but need locks on the chunks
        (i, j) = tracklets[i_trk]
        for branch in (true, false)
            pos, vel, t = project_tracklet(detections[i], detections[j], hypothesis; branch = branch)
            rho12 = norm(pos - (detections[j].pos - detections[i].pos) / (detections[j].time - detections[i].time) * (t - detections[i].time))

            if rho12 >= orbit_bounds.rho_min && isin_bounds(pos, vel, orbit_bounds, hypothesis.gm)
                pos, vel = Kepler.solve(pos, vel, hypothesis.t0 - t, hypothesis.gm)
                a, b  = vec2ang(normalize(pos)...)
                pixel = ang2pixRing(res, a, b)
                push!(healpix_partitions[pixel].points, SA[pos..., vel...])
                push!(healpix_partitions[pixel].index,  i_trk)
            end
        end
    end
end

function project_tracklet_quadratic(d1, d2, hypothesis; branch = false)
    dt1  = d1.time - hypothesis.t0
    dt2  = d2.time - hypothesis.t0
    r1   = hypothesis.r0 + hypothesis.dr0 * dt1 + 0.5hypothesis.ddr0 * dt1^2
    r2   = hypothesis.r0 + hypothesis.dr0 * dt2 + 0.5hypothesis.ddr0 * dt2^2
    rho1 = sphere_intersect_branch(d1.pos, d1.ang, r1; neg_branch = branch)
    rho2 = sphere_intersect_branch(d2.pos, d2.ang, r2; neg_branch = branch)
    pos1 = d1.ang*rho1 + d1.pos
    pos2 = d2.ang*rho2 + d2.pos
    t1   = d1.time - rho1/hypothesis.c
    t2   = d2.time - rho2/hypothesis.c

    pos, vel, t = lambert_central_diff(pos1, pos2, t1, t2)
    return pos, vel, t
end

# function project_tracklet_quadratic_converge(d1, d2, hypothesis; branch = false)
#     return pos, vel, t
# end

# function project_tracklet_analytic(d1, d2, hypothesis; branch = false)
#     return pos, vel, t
# end


# function project_tracklet_analytic_converge(d1, d2, hypothesis; branch = false)
#     return pos, vel, t
# end

function isin_bounds(pos, vel, bounds, gm; R2eccl = I3)
    zhat = SVector{3}(0.0, 0.0, 1.0)
    hvec = cross(pos, vel)
    hhat = normalize(hvec)
    hhat_eccl = R2eccl*hhat

    inc = acosd(dot(zhat, hhat_eccl) / (norm(zhat)*norm(hhat_eccl)))
    !(bounds.i_min <= inc <= bounds.i_max) && return false

    r    = norm(pos)
    v02  = dot(vel, vel)
    a    = 1/(2/r - v02/gm)
    evec = cross(vel, hvec)/gm - pos/r
    e    = norm(evec)
    rp   = a*(1 - e)

    (rp < bounds.r_min || !(bounds.e_min <= e <= bounds.e_max)) && return false

    return true
end

function lambert_central_diff(pos1, pos2, t1, t2)
    pos = 0.5(pos1 + pos2)
    t   = 0.5(t1 + t2)
    vel = (pos1 - pos2)/(t1 - t2)
    return pos, vel, t
end

function lambert_central_diff2(pos1, pos2, t1, t2, gm)
    pos = 0.5(pos1 + pos2)
    t   = 0.5(t1 + t2)
    vel = (pos1 - pos2)/(t1 - t2)
    return pos, vel, t
end

""" takes columns, rather than det structs

angles (ang) nneds to be given as cartesian unit vectors
"""
function make_arrows(
    tracklets, 
    angles, 
    times, 
    obs, 
    hypothesis, 
    orbit_bounds;
    n_threads = 1, 
    max_iters = 0,
    min_range = 0.01,
    R2ECL = I,
    tol_ddr_rel = 1e-4,
    tol_ddr_abs = 1e-6,
    # tol_ang_rate = Inf,
)
    
    n = 2length(tracklets)
    arrows = Vector{SVector{6, Float32}}(undef, n)
    trkids = Vector{Int}(undef, n)
    branch = BitArray(undef, n) 
    mask   = BitArray(undef, n) 
    mask  .= false # just in case

    # for (i, ang, t, o, ) in zip(eachindex())
    Threads.@threads for i_t in 1:n_threads
        for i in i_t:n_threads:length(tracklets) 
            (j1, j2) = tracklets[i]
            ang1 = angles[j1]
            ang2 = angles[j2]
            dt1  = times[j1] - hypothesis.t0
            dt2  = times[j2] - hypothesis.t0
            o1   = obs[j1]
            o2   = obs[j2]

            ref_r   = hypothesis.r0
            ref_dr  = hypothesis.dr0
            ref_ddr = hypothesis.ddr0

            r1 = ref_r + ref_dr*dt1 + (ref_ddr*dt1^2)/2
            r2 = ref_r + ref_dr*dt2 + (ref_ddr*dt2^2)/2
            for _branch in (false, true)
                rho1 = sphere_intersect_branch(o1, ang1, r1; neg_branch = _branch)
                rho2 = sphere_intersect_branch(o2, ang2, r2; neg_branch = _branch)

                if rho1 < min_range || rho2 < min_range
                    continue # skips
                end

                # correct for light time
                lt1 = dt1 - rho1/hypothesis.c
                lt2 = dt2 - rho2/hypothesis.c

                pos1 = o1 + rho1*ang1
                pos2 = o2 + rho2*ang2
                pos, vel, dt = lambert_central_diff(pos1, pos2, lt1, lt2)

                ii = 0
                err_ddr_abs = Inf
                err_ddr_rel = Inf
                while ii < max_iters && err_ddr_abs > tol_ddr_abs && err_ddr_rel > tol_ddr_rel
                    # update ddr0 until we ang momentum matches angular motion
                    ii += 1
                    
                    hvec    = cross(pos, vel)
                    h2      = dot(hvec, hvec) # h^2
                    ddr_est = h2/ref_r^3 - hypothesis.gm/ref_r^2

                    err_ddr_abs = abs(ddr_est - ref_ddr)
                    err_ddr_rel = abs(err_ddr_rel/ref_ddr)
                    ref_ddr = ddr_est
                end

                # check orbit against cut offs, propagate, add to arrows
                if isin_bounds(pos, vel, orbit_bounds, hypothesis.gm; R2eccl = R2ECL)
                    pos, vel = Kepler.propagate(pos, vel, -dt, hypothesis.gm)
                    j = 2i - _branch
                    arrows[j] = SA[pos..., vel...]
                    trkids[j] = i
                    branch[j] = _branch
                    mask[j]   = true
                end
            end
        end
    end

    return arrows[mask], trkids[mask], branch[mask]
end

# function project_tracklet_old(
#     d1, d2, 
#     r0, dr0, t0, gm; 
#     c = Inf, 
#     neg_branch = false, 
#     state_estimate = CentralDifference1,
#     distance_model = QuadraticDistance,
#     max_iter = 10, 
#     rho_min = 0.0, 
#     rel_tol = 1e-12, abs_tol = 1e-12
#     )
#     dt1   = d1.time - t0
#     dt2   = d2.time - t0
#     ddr0  = 0.0
#     alpha = 1/r0 - dr0^2/gm # initial guess

#     # check for valid transverse motion
#     if (2/r0 - alpha)*gm - dr0^2 <= 0
#         return nothing
#     end

#     # instantiate 
#     pos = zero(d1.pos)
#     vel = zero(d1.vel)
#     t   = zero(d1.time)

#     i = 0
#     rel_err = Inf
#     abs_err = Inf
#     while i < max_iter && rel_err > rel_tol && abs_err > abs_tol
#         (r1, r2) = if distance_model == LinearDistance
#             r1 = r0 + dr0*dt1
#             r2 = r0 + dr0*dt2
#             (r1, r2)
#         elseif distance_model == QuadraticDistance
#             r1 = r0 + dr0*dt1 + 0.5ddr0 * dt1^2
#             r2 = r0 + dr0*dt2 + 0.5ddr0 * dt2^2
#             (r1, r2)
#         elseif distance_model == AnalyticModel
#             r1 = propagate_radial_distance(r0, dr0, alpha, dt1, gm)
#             r2 = propagate_radial_distance(r0, dr0, alpha, dt2, gm)
#             (r1, r2)
#         else
#             throw("invalid distance mdoel specified")
#         end

#         xyz1 = SVector{3}(sphere2cart(d1.ang...))
#         xyz2 = SVector{3}(sphere2cart(d2.ang...))
        
#         rho1 = sphere_intersect_branch(d1.pos, xyz1, r1; neg_branch = neg_branch)
#         rho2 = sphere_intersect_branch(d2.pos, xyz2, r2; neg_branch = neg_branch)

#         if rho1 < rho_min || rho2 < rho_min # invalid state
#             return nothing
#         end

#         pos1 = rho1*xyz1 + d1.pos
#         pos2 = rho2*xyz2 + d2.pos
#         tl1  = d1.time - rho1/c
#         tl2  = d2.time - rho2/c

#         pos, vel, t = if state_estimate == CentralDifference1
#             estimate_state(pos1, pos2, tl1, tl2)
#         elseif state_estimate == CentralDifference2
#             estimate_state_2(pos1, pos2, tl1, tl2, gm)
#         else
#             throw("unimplemetned method specified")
#         end

#         r  = norm(pos)
#         v2 = dot(vel, vel)
#         hvec = cross(pos, vel)
#         h2   = dot(hvec, hvec)
#         ddr0 = h2/r^3 - gm/r^2 
#         alpha2  = 2/r - v2/gm
#         abs_err = abs(alpha - alpha2)
#         rel_err = abs(abs_err / alpha)
#         alpha   = alpha2

#         if (2/r0 - alpha)*gm - dr0^2 <= 0
#             return nothing
#         end
#         i += 1
#     end
#     if i >= max_iter
#         return nothing
#     end
#     return pos, vel, t
# end

# function propagate_radial_distance(r0, dr0, alpha, dt, gm)
#     vt   = sqrt((2/r0 - alpha)*gm - dr0^2)
#     pos0 = SVector{3}((r0, 0.0, 0.0))
#     vel0 = SVector{3}((dr0, vt, 0.0))
#     try
#         posf, _ = Kepler.solve(pos0, vel0, dt, gm)
#         return norm(posf)
#     catch e
#         println("$(pos0) $(vel0), $(dt), $(gm)")
#     end
# end

# function estimate_state(pos1, pos2, t1, t2)
#     pos = 0.5(pos1 + pos2)
#     t   = 0.5(t1 + t2)
#     vel = (pos1 - pos2)/(t1 - t2)
#     return pos, vel, t
# end

# function estimate_state_2(pos1, pos2, t1, t2, gm)
#     dt = t2 - t1
#     acc1 = -gm*pos1 / norm(pos1)^3
#     acc2 = -gm*pos2 / norm(pos2)^3
#     vel1 = (pos2 - pos1)/dt - (2acc1 + acc2)*dt/6
#     vel2 = (pos2 - pos1)/(-dt) - (2acc2 + acc1)*(-dt)/6
#     pos121 = pos1 + vel1*dt/2 + 0.5acc1*(dt/2)^2
#     vel121 = vel1 + acc1*dt/2
#     pos122 = pos2 - vel2*dt/2 + 0.5acc2*(dt/2)^2
#     vel122 = vel2 - acc2*dt/2
#     pos    = 0.5(pos121 + pos122)
#     vel    = 0.5(vel121 + vel122)
#     t = 0.5(t1 + t2)
#     return pos, vel, t
# end

# function compute_ddr0(pos, vel, r0, gm)
#     h = norm(cross(pos, vel))
#     return h^2/r0^3 - gm/r0^2
# end
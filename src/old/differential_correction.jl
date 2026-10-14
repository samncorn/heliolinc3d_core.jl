""" state vector convenience getters
"""
# get_position(x::SVector{6, T}) where {T} = SVector{3, T}(view(x, 1:3))
# get_velocity(x::SVector{6, T}) where {T} = SVector{3, T}(view(x, 4:6))
get_position(x::SVector{6, T}) where {T} = SVector{3, T}(x[1], x[2], x[3])
get_velocity(x::SVector{6, T}) where {T} = SVector{3, T}(x[4], x[5], x[6])

function apparent_state(target_traj, obs_pos, t, c; lt_tol = 1e-15, max_iter = 20)
    lt = 0.0
    dt = Inf
    i  = 0
    while dt > lt_tol && i < max_iter
        i += 1
        # println(i)
        # println(" ti = $(t - lt) ")
        lti = try
            pos = target_traj(t - lt)
            if any(isnan.(pos))
                throw((dt = t - lt))
            end
            norm(pos - obs_pos) / c
        catch _
            throw((dt = t-lt,))
        end
        dt  = abs(lt - lti)
        lt = lti
    end

    # println("evaluating light time corrected state")
    pos_app = try
        target_traj(t - lt) - obs_pos
    catch _
        throw((dt = t-lt,))
    end

    return pos_app, t - lt, i
end

function state_to_angles(x0, dt, obs_pos, gm, c)
    traj = t -> Kepler.propagate(get_position(x0), get_velocity(x0), t, gm)[1]
    app  = apparent_state(traj, obs_pos, dt, c; max_iter = 3)[1]
    r    = norm(app)
    xhat = app/r
    return xhat
end

function state_to_angles_with_partials(x0, dt, obs_pos, gm, c)
    traj(t)    = Kepler.propagate(get_position(x0), get_velocity(x0), t, gm)[1]
    # println("computing apparent state")
    # println("  pos = $(get_position(x0))")
    # println("  vel = $(get_velocity(x0))")
    # println("  dt  = $dt")
    # println("  gm  = $gm")
    app, lt, _ = apparent_state(traj, obs_pos, dt, c; max_iter = 3)

    # get the partials
    # println("computing partials")
    # _, _, dxdx, dxdv, _, _ = Kepler.propagate_with_partials(get_position(x0), get_velocity(x0), lt, gm)
    _, _, stm = Kepler.propagate_stm(get_position(x0), get_velocity(x0), lt, gm)
    dxdx = stm.dX_dX0
    dxdv = stm.dX_dV0
    
    r     = norm(app)
    xhat  = app/r
    dxhdx = (I3 - xhat*transpose(xhat))/r
    
    J1 = dxhdx*dxdx
    J2 = dxhdx*dxdv
    J  = hcat(J1, J2)

    return xhat, J
end

""" computes o-c in the tangent plane by projecting c onto the o tangent
"""
function project_angle_resids(o_ang, c_ang)
    # get our tangent plane basis vectors
    z = SVector{3}(0.0, 0.0, 1.0)
    a = normalize(cross(z, o_ang))
    d = normalize(cross(o_ang, a))

    return -SVector{2}(dot(c_ang, a), dot(c_ang, d))
end

function project_angle_resids_with_partials(o_ang, c_ang)
    # get our tangent plane basis vectors
    z = SVector{3}(0.0, 0.0, 1.0)
    a = normalize(cross(z, o_ang))
    d = normalize(cross(o_ang, a))

    # partials
    J = transpose(hcat(a, d))

    return -SVector{2}(dot(c_ang, a), dot(c_ang, d)), -J
end

# """ fit a keplerian 2-body orbit to a set of angular measurements

# ra, dec in radians
# uncertainties in radians
# """
# function fit_kepler(ra, dec, ra_sigma, dec_sigma, times, obs, pos0, vel0, t0, gm; c = Inf, tol = 1e-8, max_iter = 10, sink = nothing, mu = 0.0)
#     angles  = [SVector{3}(sphere2cart(ai, bi)) for (ai, bi) in zip(ra, dec)]
#     weights = [SMatrix{2, 2}(1/sa^2, 0.0, 0.0, 1/sb^2) for (sa, sb) in zip(ra_sigma, dec_sigma)]
    
#     return fit_kepler(angles, weights, times, obs, pos0, vel0, t0, gm; c = c, tol = tol, max_iter = max_iter, sink = sink, mu = mu)
# end

""" fit a keplerian orbit
------
angles ... Vector{SVector{3}}
    cartesian unit vectors indicating the pointing angles
weights ... Vector{SMatrix{2, 2}}
    weight matrices for each measurement vector
"""
function fit_kepler_analytic(angles, weights, times, obs, pos0, vel0, t0, gm, c; tol = 1e-10, max_iter = 10, sink = nothing, mu = 0.0)
    fdf = x -> Iterators.map(((t, o, ang),) -> fit_kepler_resids_with_partials_analytic(x, t-t0, o, ang, gm, c), zip(times, obs, angles))
    x0  = reduce(vcat, (pos0, vel0))

    return levenberg_marquardt(fdf, x0, weights; mu = mu, tol_cost_rel = tol, tol_cost_abs = tol, tol_param_rel = tol, tol_param_abs = tol, max_iter = max_iter, sink = sink)
end

function fit_kepler_auto(angles, weights, times, obs, pos0, vel0, t0, gm, c; tol = 1e-10, max_iter = 10, sink = nothing, mu = 0.0)
    fdf = x -> Iterators.map(((t, o, ang),) -> fit_kepler_resids_with_partials_auto(x, t-t0, o, ang, gm, c), zip(times, obs, angles))
    x0  = reduce(vcat, (pos0, vel0))

    return levenberg_marquardt(fdf, x0, weights; mu = mu, tol_cost_rel = tol, tol_cost_abs = tol, tol_param_rel = tol, tol_param_abs = tol, max_iter = max_iter, sink = sink)
end

function fit_kepler_resids_with_partials_auto(x, dt, o, ang, gm, c) 
    _func(_x) = fit_kepler_resids(_x, dt, o, ang, gm, c)
    resids = _func(x)
    J      = ForwardDiff.jacobian(_func, x)

    return resids, -J
end

function fit_kepler_resids_with_partials_analytic(x, dt, o, ang, gm, c)
    angc,  J1 = state_to_angles_with_partials(x, dt, o, gm, c)
    resid, J2 = project_angle_resids_with_partials(ang, angc)
    return resid, -J2*J1
end

fit_kepler_resids(x, dt, o, ang, gm, c) = project_angle_resids(ang, state_to_angles(x, dt, o, gm, c))

# "ra, dec should be in RADIANS"
# function differential_correction(pos0, vel0, t0, times, ra, dec, obs_pos, gm; c = Inf, del = 1e-8, tol = 1e-8, max_iter = 10)
#     @assert length(times) == length(ra) == length(dec) == length(obs_pos)
#     # set up resids and partials vectors
#     resids   = Vector{SVector{2, Float64}}(undef, length(times))
#     partials = Vector{SMatrix{2, 6, Float64}}(undef, length(times)) # 2x6?
#     x0 = SVector{6}(vcat(pos0, vel0))
#     # run levenberg marquardt
#     # f(x)  = od_resids!(resids, x, t0, times, ra, dec, obs_pos, gm; c = c)
#     # df(x) = od_partials_auto!(partials, x, t0, times, ra, dec, obs_pos, gm; c = c, del = del)
#     fdf = x -> od_resids_partials_auto!(resids, partials, x, t0, times, ra, dec, obs_pos, gm; c = c)
#     return levenberg_marquardt(fdf, x0; tol_rel = tol, tol_abs = tol, max_iter = max_iter), resids
# end

# function od_resids!(resids, x0, t0, times, ra, dec, obs_pos, gm; c = Inf)
#     @assert length(resids) == length(times) == length(ra) == length(dec) == length(obs)
#     traj(t) = Kepler.propagate(get_position(x0), get_velocity(x0), t-t0, gm)[1]
#     for (i, t, a, b, o) in zip(eachindex(resids), times, ra, dec, obs_pos)
#         # get apparent state
#         pos_app, _, _ = apparent_state(traj, o, t, c; max_iter = 3)
#         # convert to ra, dec
#         ang       = SVector{3}(sphere2cart(a, b))
#         resids[i] = compute_angles_resid(pos_app, ang)
#     end
#     return resids
# end

# function od_partials_central_diff!(partials, x0, t0, times, ra, dec, obs_pos, gm; c = Inf, del = 1e-8)
#     for (i, t, a, b, o) in zip(eachindex(resids), times, ra, dec, obs_pos)
#         for j in 1:eachindex(x0)
#             dx     = @MVector zeros(6)
#             dx[j] += del
#             x_u    = x0 + dx
#             x_d    = x0 - dx
#             traj_u = Kepler.propagate(get_position(x_u), get_velocity(x_u), t-t0, gm)[1]
#             traj_d = Kepler.propagate(get_position(x_d), get_velocity(x_d), t-t0, gm)[1]

#             pos_app_u = apparent_state(traj_u, o, t, c; max_iter = 3)
#             pos_app_d = apparent_state(traj_d, o, t, c; max_iter = 3)

#             ang = SVector{3}(sphere2cart(a, b))
#             resid_u = compute_angles_resid(pos_app_u, ang)
#             resid_d = compute_angles_resid(pos_app_d, ang)
#             partials[i][1:2, j] .= (resid_u + resid_d) / (2del)
#         end
#         partials[i] = SMatrix{2, 6}(Jac)
#     end
#     return partials
# end

# function od_resids_partials_auto!(resids, partials, x0, t0, times, ra, dec, obs_pos, gm; c = Inf)
#     for (i, t, a, b, o) in zip(eachindex(resids), times, ra, dec, obs_pos)
#         # here we convert ra, dec to angles vectors. we could do this berfore hand and just pass the vectors
#         ang = SVector{3}(sphere2cart(a, b; deg=true))
#         resid_func  = x -> state_to_resid(x, t-t0, o, ang, gm, c)
#         resids[i]   = resid_func(x0)
#         partials[i] = ForwardDiff.jacobian(resid_func, x0)
#     end
#     return resids, partials
# end
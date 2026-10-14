
# dont worry about precaching atm. 
# """ Store an interpolant of the trajectory and error stm

# 3 collections are maintained. One has elements of 42-long sized vectors, to store the results of integration of the 
# 6 + 36 state and stm elements. The second is a vector of sized stm matrices to be interpolated, for convenience 
# downstream, this avoids having to reshape a matrix every time the interpolation is called. The third is similarly the 
# sized state vectors.
# """
# struct ODCache{T}
#     traj_stm   :: Vector{SVector{42, T}}
#     stm        :: Vector{SMatrix{6, 6, T}}
#     trajectory :: Vector{SVector{6, T}}
# end

# function IODCache(T, )

# need obs to have time and angles fields
function herget_iod(
    obs::Vector{AnglesMeas{T}},
    i::Int, 
    j::Int,
    rhoij0::SVector{2, T}, 
    dynamics::Function, 
    p, 
    c::T,
    ; 
    sink    = nothing,
    tol_rel = 1e-10, 
    tol_abs = 1e-11, 
    mu_f    = 100.0,
    ) where {T}

    n_obs = length(obs)
    obsij = (obs[i], obs[j])
    v_resids   = Vector{SVector{2, T}}(undef, n_obs)
    v_partials = Vector{SMatrix{2, 2, T}}(undef, n_obs)

    fdf(x) = integrate_herget_resids!(v_resids, v_partials, x, obsij, obs, dynamics, p, c)
    # least squares
    # rhoijf, resids = _levenberg_marquardt(fdf, rhoij0;
    #     sink = sink, 
    #     tol_rel = tol_rel, 
    #     tol_abs = tol_abs, 
    #     mu_f = mu_f,
    #     )
    # # return fit params and residuals
    # return rhoijf, resids
    result = levenberg_marquardt(
        fdf, rhoij0;
        sink = sink, 
        tol_rel = tol_rel, 
        tol_abs = tol_abs, 
        mu_f = mu_f,
    )
    return result, v_resids
end

""" Uses an unscented transform to compute the partial derivatives of the residuals w.r.t. parameters 
"""
# function unscented_herget_resids!(v_resids, v_partials, rhoij, obsij, obs, gm; c = Inf, del = 1e-4)
#     @assert length(v_resids) == length(v_partials) == length(obs)
#     times = Iterators.map(o -> o.time, obs)

#     # estimate the state (light time corrected)
#     pos1 = 
#     pos2 = 
#     t1   = 
#     t2   = 
#     dt   = t2 - t1
#     vel1 = Kepler.Lambert.solve(pos1, pos2, dt, gm; Kepler.Lambert.High, Kepler.Lambert.Short)

#     # choose sigma points
#     rhoij11 = rhoij + SA[ del,  del]
#     rhoij12 = rhoij + SA[ del, -del]
#     rhoij21 = rhoij + SA[-del,  del]
#     rhoij22 = rhoij + SA[-del, -del]

#     # cache and interpolate trajectory?

#     # for each time, obs,
#     for (i, o) in enumerate(obs)
#         # get apparent state, corrected time
#         traj(t) = Kepler.solve(pos1, vel, t-t1, gm)
#         posi, veli, ti = _apparent_state(traj, o.pos, o.vel, o.time, c; max_iter = 3,)
#         # map sigma points to time and measurement

#         # compute partials
        
#     return v_resids, v_partials
# end

""" compute residuals and partials for the iod method of herget using keplerian dynamics, and some autodiff.
"""
function herget_resids_partials!(resids, partials, ranges, idx, times, ra, dec, obs_pos, gm, c)
    @assert length(resids) == length(partials) == length(times) == length(ra) == length(dec) == length(obs)
    # convert ranges to initial states (plus jacobian)
    i1, i2 = idx
    ang1 = sphere2cart(ra[i1], dec[i1]; deg = true)
    ang2 = sphere2cart(ra[i2], dec[i2]; deg = true)
    x0   = estimate_state(ranges, times[i1], times[i2], ang1, ang2, obs_pos[i1], obs_pos[i2], gm, c)
    J1   = ForwardDiff.jacobian(x -> estimate_state(x, times[i1], times[i2], ang1, ang2, obs_pos[i1], obs_pos[i2], gm, c), ranges) 

    for (i, t, a, b, o) in zip(eachindex(resids), times, ra, dec, obs_pos)
        dt = t - times[i1]
        resid = state_to_resid(x0, dt, )

    end
    return resids, partials
end

""" times as observation times"""
function estimate_state(ranges, t1, t2, ang1, ang2, obs1, obs2, gm, c)
    p1, p2 = ranges
    pos1 = p1*ang1 + obs1
    pos2 = p2*ang2 + obs2

    lt1 = t1 - p1/c
    lt2 = t2 - p2/c
    vel1, _ = Kepler.lambert_solve(pos1, pos2, lt2 - lt1, gm)

    return SVector{6}(vcat(pos1, vel1))
end

"""
"""
function state_to_resid(x0, dt, obs_pos, ang, gm, c)
    traj   = t -> Kepler.propagate(get_position(x0), get_velocity(x0), t, gm)[1]
    posapp = apparent_state(traj, obs_pos, dt, c; max_iter = 3)[1]
    return compute_angles_resid(posapp, ang)
end

# function herget_partials!()

""" integrates both the trajectory and the stm, returns vectors of residuals and partials

dynamics and p need to conform to the ODEProblem format
"""
function integrate_herget_resids!(v_resids, v_partials, rhoij, obsij, obs, dynamics, p, c; integ = Tsit5(), time_buffer = 1.0)
    @assert length(v_resids) == length(v_partials) == length(obs)
    times = Iterators.map(o -> o.time, obs)
    t_min = minimum(times) - time_buffer
    t_max = maximum(times) + time_buffer
    # choose the first obs for no particular reason. also light time correct it
    t0 = obsij[1].time - rhoij[1]/c 
    
    # estimate initial state, get params -> state partials
    rho2pv(x)  = estimate_state(x, obsij, dynamics, p, c; integ = integ)
    # (J1,), pv0 = Enzyme.gradient(ForwardWithPrimal, rho2pv, rhoij)
    # doesnt work with ezyme, could try forwarddiff
    # but as it turns out, we can compute the initial state sensitivity to range changes with the stm
    # so if we first solve the lambert problem, we can compute the range partials with stuff we compute anyways
    pv0 = estimate_state(rhoij, obsij, dynamics, p, c; integ = integ)
    J1  = ForwardDiff.jacobian(rho2pv, rhoij) 

    # integrate state + stm
    # if needed, glue togther two trajectories
    # in fact, it has to be needed if the time buffer is not zero (which is need for c < inf)
    # traj_stm = if t0 < t_min
    tspan1 = (t0, t_min)
    tspan2 = (t0, t_max) # backwards integration
    traj_stm1 = integrate_traj_stm(dynamics, pv0, tspan1, p; integ = integ)
    traj_stm2 = integrate_traj_stm(dynamics, pv0, tspan2, p; integ = integ)
    # and now we glue it together 
    traj_stm(t) = t < t0 ? traj_stm1(t) : traj_stm2(t)
    # need just the position for finding the apparent state
    traj(t) = SVector{3}(view(traj_stm(t), 1:3))

    for (i, o) in enumerate(obs)
        # get light time corrected state, time
        # per the SPICE docs on light time correction, only a few iterations are needed to guaruntee sufficient accuracy
        # this prevents the error from propagating downstream
        pos_app, t_lt, _ = _apparent_state(traj, o.pos, o.vel, o.time, c; max_iter = 3,) # stellar_aberration = true)
        # get the stm
        J2 = SMatrix{6, 6}(view(traj_stm(t_lt), 7:42))
        # compute the observable + partials
        app2res(x) = compute_angles_resid(x, o.ang) 
        (J3,), res_ci = Enzyme.jacobian(ForwardWithPrimal, app2res, pos_app)
        J3 = hcat(J3, @SMatrix zeros(2, 3))
        v_resids[i]   = res_ci
        v_partials[i] = J3*J2*J1 # chain the partials
    end

    return v_resids, v_partials
end

function integrate_traj_stm(dynamics, pv0, tspan, p; integ = Tsit())
    # initial state + stm
    stm0 = I6
    u0   = SA[pv0..., stm0...]

    # state + stm dynamics
    stm_dynamics(u, p, t) = stm_dynamics_auto(dynamics, u, p, t)
    traj = integrate_traj(stm_dynamics, u0, tspan, p, integ)
    return traj
end

"""
returns the integrated interpolant

this is an extremely light wrapper around the differential equations functions, so it maintains most of the behavior.
ie, for the timespan, the first element is the time of the initial conditions, and the second is the time to stop at, 
which can be less than the first time for backwards integration. the returned interpolant then takes times according to 
the given timespan, not times relative to the initial state

so if you have an initial condition in the middle of a timespan, you need to either propagate to one of the end points,
then do a second integration oiver the timespan, or in both direction and glue the solutions together. 

bunch of check for valid integrations
"""
function integrate_traj(dynamics, pv0, tspan, p, integ; kwargs...)
    # sanity checks on inputs
    @assert any(.!isnan.(pv0))
    @assert any(.!isnan.(tspan))
    @assert any(.!isnan.(p))

    ode = ODEProblem(dynamics, pv0, tspan, p)
    sol = solve(ode, integ; kwargs...)
    return sol
end

function stm_dynamics_auto(dynamics, u, p, t)
    pv   = SVector{6}(view(u, 1:6))
    stm  = SMatrix{6, 6}(view(u, 7:42))
    f(x) = dynamics(x, p, t)
    (dphi,), dpv = Enzyme.jacobian(ForwardWithPrimal, f, pv)
    dstm = dphi * stm
    return SVector{42}(vcat(dpv, dstm...))
end

""" frame needs to be alligned with the angles system
"""
function compute_angles_resid(pos_app, ang) 
    ang_c = normalize(pos_app)
    z_hat = SVector{3}((0.0, 0.0, 1.0))
    # obs projection vectors
    a_hat = normalize(cross(z_hat, ang))
    d_hat = cross(ang, a_hat)
    return SVector{2}((dot(ang_c, a_hat), dot(ang_c, d_hat)))
end

"""
terminates after either meeting tolerance or reacing max_iter. Throws an error for max_iter

traj needs to return the full pos, vel state
"""
function _apparent_state(target_traj, obs_pos, obs_vel, t, c; lt_tol = 1e-15, max_iter = 20, stellar_aberration = false)
    lt = 0.0
    dt = Inf
    i  = 0
    while dt > lt_tol && i < max_iter
        lti = norm(target_traj(t - lt) - obs_pos) / c
        dt  = abs(lt - lti)

        lt = lti
        i += 1
    end

    # apply stellar aberration
    pos_app = target_traj(t - lt) - obs_pos

    if stellar_aberration
        sinw = dot(normalize(obs_vel), normalize(pos_app))
        phi  = asin(norm(obs_vel)*sinw/c)

        mom = cross(obs_pos, obs_vel)

        R = AngleAxis(phi, mom...)
        pos_app = R*pos_app
    end

    return pos_app, t - lt, i
end

function apparent_state(target_traj, obs_pos, t, c; lt_tol = 1e-15, max_iter = 20)
    lt = 0.0
    dt = Inf
    i  = 0
    while dt > lt_tol && i < max_iter
        lti = norm(target_traj(t - lt) - obs_pos) / c
        dt  = abs(lt - lti)

        lt = lti
        i += 1
    end

    # apply stellar aberration
    pos_app = target_traj(t - lt) - obs_pos

    return pos_app, t - lt, i
end

# function apparent_state(target_traj, observer_pos, observer_vel, t, c; lt_tol = 1e-12, max_iter = 20)
#     state, lt, i = _apparent_state(target_traj, observer_pos, observer_vel, t, c; lt_tol = lt_tol, max_iter = max_iter)
#     # if i >= max_iter
#     #     @debug("apparent state failed after $(i) iters with tol $(lt_tol)")
#     #     @debug("t: $(t), dlt: $(t - lt)")
#     #     @debug("c: $(c)")
#     #     throw(MaximumIterationError(i))
#     # end
#     return state, lt
# end

""" Solve the lambert problem numerically (shooting method) for the single revolution solution

dynamics, p needs to conform to the diffeq format
"""
function _lambert_dns(pos1, pos2, dt, dynamics, p; max_iter = 20, abstol = 1e-15, reltol = 1e-15, integ = Tsit5())
    # finite diff the velocity for initial guess
    vel1 = (pos2 - pos1)/dt
    pv1  = SA[pos1..., vel1...]
    iter = 1
    rel_err  = Inf
    abs_err  = Inf
    while iter < max_iter && abs_err > abstol && rel_err > reltol
        # posf, _  = propagate(pos1, vel1, dt)
        ode = ODEProblem(dynamics, pv1, (0.0, dt), p)
        sol = solve(ode, integ; abstol = abstol, reltol = reltol, save_everystep = false, save_start = false)
        posf = SA[view(sol.u[end], 1:3)...]

        dvel  = (pos2 - posf)/dt
        vel1 += dvel
        pv1   = SA[pos1..., vel1...]
        abs_err = norm(dvel)
        rel_err = norm(dvel)/norm(vel1)
        iter += 1
    end
    return iter, vel1
end

function lambert_dns(pos1, pos2, dt, dynamics, p; max_iter = 20, abstol = 1e-15, reltol = 1e-15, integ = Tsit5())
    i, vel = _lambert_dns(pos1, pos2, dt, dynamics, p; max_iter = max_iter, abstol = abstol, reltol = reltol, integ = integ)
    i >= max_iter && throw(MaximumIterationError(i))
    return vel
end

function estimate_state(rhoij, obsij, dynamics, p, c; integ = Tsit5())
    posi = rhoij[1] * obsij[1].ang + obsij[1].pos
    posj = rhoij[2] * obsij[2].ang + obsij[2].pos
    # light time correct the state estimate
    t1 = obsij[1].time - rhoij[1]/c
    t2 = obsij[2].time - rhoij[2]/c
    dt = t2 - t1
    veli = lambert_dns(posi, posj, dt, dynamics, p; integ = integ, reltol = 1e-12, abstol = 1e-12)
    return SA[posi..., veli...]
end

# function integrate_2b(pos, vel, dt, gm)
#     u0  = SA[pos..., vel...]
#     ode = ODEProblem(dynamics_2b, u0, (0.0, dt), gm)
#     sol = solve(ode, Tsit5(), abstol = 1e-12, reltol = 1e-12, save_everystep = false, save_start = false).u[end]
#     return SVector{3}(view(sol, 1:3)), SVector{3}(view(sol, 4:6))
# end

# function dynamics_2b(u, p, _)
#     pos = SVector{3}(view(u, 1:3))
#     vel = SVector{3}(view(u, 4:6))
#     acc = -p/norm(pos)^3 * pos
#     return SVector{6}((vel..., acc...))
# end

# function dynamics(pv, f)
#     vel = SVector{3}(view(u, 4:6))
#     acc = f(pv)
#     return SVector{6}((vel..., acc...))
# end
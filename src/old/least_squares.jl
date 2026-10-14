# helper for (not) logging steps
import Base.push!

push!(::Nothing, _) = nothing

""" residuals need to be given as (observed - computed)

Weighted, so fdf should return (resid, partials, weights) for each measurement. If measurements are 
"""
function levenberg_marquardt(fdf, x0, weights; mu = 100.0, mu_f = 5.0, tol_cost_rel = 1e-8, tol_cost_abs = 1e-8, tol_param_rel = 1e-8, tol_param_abs = 1e-8, max_iter = 100, sink = nothing)
    x = x0
    i = 0
    step_rel_cost  = Inf
    step_abs_cost  = Inf
    step_rel_param = Inf
    step_abs_param = Inf
    cost, Hty, HtH = _weighted_least_squares_kernel(fdf, weights, x)
    push!(sink, (
        i = i, 
        x = x, 
        dx = Inf, 
        cost = cost, 
        mu = mu, 
        step_abs_cost = step_abs_cost,
        step_rel_cost = step_rel_cost,
        step_abs_param = step_abs_param,
        step_rel_param = step_rel_param,
        ))
    while (i < max_iter 
        && step_rel_cost > tol_cost_rel 
        && step_abs_cost > tol_cost_abs 
        && step_rel_param > tol_param_rel 
        && step_abs_param > tol_param_abs
        # && norm(x - x0)/norm(x0) < 1e3
        )
        i += 1
        # println(i)
        dx = try
            pinv(HtH + mu*I)*Hty
        catch _
            throw(throw((iteration = i, state = x)))
        end
        new_cost, Hty, HtH = try 
            _weighted_least_squares_kernel(fdf, weights, x + dx)
        catch _
            throw((iteration = i, state = x + dx))
        end

        # println("computed step")

        if new_cost >= cost
            # increase gradient step
            mu *= mu_f
            # continue
        else
            # advance
            mu /= mu_f
        end

        # update estimate
        x  += dx
        step_abs_cost  = abs(cost - new_cost)
        step_rel_cost  = step_abs_cost / abs(cost)
        step_abs_param = norm(dx)
        step_rel_param = step_abs_param/norm(x)
        cost    = new_cost

        # println("moved by step")

        push!(sink, (
            i = i, 
            x = x, 
            dx = dx, 
            cost = cost, 
            mu = mu, 
            step_abs_cost = step_abs_cost,
            step_rel_cost = step_rel_cost,
            step_abs_param = step_abs_param,
            step_rel_param = step_rel_param,
            )
        )
    end

    # cost is the X^2 of the residuals
    return x, cost, i
end

""" convenience function wrapping levenberg marquardt around seperate function calls for f and df
"""
function levenberg_marquardt(f, df, x0, weights; kwargs...)
    return levenberg_marquardt(x -> (f(x), df(x)), x0, weights; kwargs...)
end

# # efficientely compute the least squares matrices
# # efficient requires efficient implementation in fdf (vector or iterator of (resid, partial))
# function _least_squares_kernel(fdf, x::SVector{D, T}) where {D, T}
#     J   = zero(T)
#     Hty = @SVector zeros(T, D)
#     HtH = @SMatrix zeros(T, D, D)
    
#     # accumulate the matrices
#     for (dyi, Hi) in fdf(x)
#         # dyi  = (yi - yci)
#         J   += 0.5*dot(dyi, dyi)
#         # weights?
#         Ht   = transpose(Hi) 
#         Hty += Ht*dyi
#         HtH += Ht*Hi
#     end

#     return J, Hty, HtH
# end

""" squared weights (if statistically informed, measurment reciprocal variances)
"""
function _weighted_least_squares_kernel(fdf, weights, x::SVector{D, T}) where {D, T}
    J   = zero(T)
    Hty = @SVector zeros(T, D)
    HtH = @SMatrix zeros(T, D, D)
    
    # accumulate the matrices
    # for (dyi, Hi, wi) in fdfw(x)
    for ((dyi, Hi), wi) in zip(fdf(x), weights)
        # dyi  = (yi - yci) # o - c
        J   += dot(dyi, wi*dyi)
        # weights?
        Ht   = transpose(Hi)
        Hty += Ht*wi*dyi
        HtH += Ht*wi*Hi
    end

    return J, Hty, HtH
end
function RANSAC_OD(ra, dec, times, img_ids, ra_sigma, dec_sigma, init_guess::Kepler.Cartesian, n_samples)
    # set up groups based on img
    img_grps = Dict{Int, Vector{Int}}()
    for (d, img) in enumerate(img_ids)
        if haskey(img_grps, img)
            push!(img_grps[img], d)
        else
            img_grps[img] = [d]
        end
    end
    # ransac loop
    for _ in 1:n_samples
        # draw one meas from each image (ignore tracklets)
        # fit these meas (ignoring all others)
        # compute total residuals
        # count inliers and outliers
    end
    
end

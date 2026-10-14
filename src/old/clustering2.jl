function DBSCAN(
    partitions::Vector{hp_partition{SVector{6, T}}},
    res,
    radius,
    min_pts;
    n_threads = 1,
    metric = Euclidean()
    ) where {T}

    @assert length(partitions) == res.numOfPixels

    # build the trees
    trees = Vector{Any}(undef, res.numOfPixels)
    Threads.@threads for i_t in 1:n_threads
        for i in i_t:n_threads:length(trees)
            trees[i] = KDTree(partitions[i].points, metric)
        end
    end

    # query the disc of pixels within clustering radius
    # disc query returns pixels whose CENTERS are within a dsitance of a point, so we need to query cluster radius + pixel radius
    pixel_radius = sqrt(nside2pixarea(res.nside))
    # TODO: if cluster radius comparable to pixel radius, throw an error

    n_points = reduce(+, Iterators.map(x -> length(x.points), partitions))
    labels   = zeros(Int, n_points)
    mergers  = [Tuple{Int, Int}[] for _ in 1:res.numOfPixels]

    Threads.@threads for i_p in 1:res.numOfPixels
        dbscan_hp_kernel!(labels, mergers, trees, partitions, res, i_p, radius, pixel_radius, min_pts)
    end

    # merger clean up sweep
    for (i, j) in Iterators.flatten(mergers)
        dbscan.join_labels!(labels, i, j)
    end

    return labels
end

function dbscan_hp_kernel!(labels, mergers, trees, partitions, res, i_p, radius, pixel_radius, min_pts)
    # get neighboring pixels
    a, b   = pix2angRing(res, i_p)
    pixels = queryDiscRing(res, a, b, pixel_radius + radius)
    # allocate neighborhood vector for each pixel
    v = [Int[] for _ in pixels]

    for (li, p_i) in zip(partitions[i_p].indices, partitions[i_p].points)
        load_neighborhoods!(v, partitions, pixels, trees, p_i, radius)

        # do the actual dbscan stuff
        n_neighbors = reduce(+, Iterators.map(length, v))
        # need equality because the point will count itself
        n_neighbors <= min_pts && continue

        # if not flagged yet, promote to cluster
        labels[li] == 0 && (labels[li] = li)

        # label points in the neigborhood
        label_points!(labels, mergers, v, pixels, li, i_p)
    end
end

function load_neighborhoods!(v, partitions, pixels, trees, p_i, radius)
    # load each neighborhood
    for (vp, pix) in zip(v, pixels)
        empty!(vp)
        inrange!(vp, trees[pix], p_i, radius)
        # promote local index to global index
        for (i, j) in enumerate(vp)
            vp[i] = partitions[pix].indices[j]
        end
    end
end

function label_points!(labels, mergers, v, pixels, li, i_p)
    for (vp, pix) in zip(v, pixels)
        for lj in vp
            if pix == i_p
                dbscan.join_labels!(labels, li, lj)
            else
                push!(mergers[i_p], (li, lj))
            end
        end
    end
end

function collect_labels(labels::AbstractVector{T}) where T
    clusters    = Dict{Int, Vector{T}}()
    for (i, l) in pairs(labels)
        if haskey(clusters, l)
            # clusters[l] = [i]
            push!(clusters[l], i)
        else
            clusters[l] = [i]
            # push!(clusters[l], i)
        end
    end
    return clusters
end

# function collect_labels(labels)
#     cluster_ids = unique(labels)
#     clusters    = Dict{Int, Vector{Int}}()
#     Threads.@threads for i_t in 1:n_threads
#         for i in i_t:n_threads:length(cluster_ids)
#             li = cluster_ids[i]
#             n = 0
#             for lj in labels
#                 li == lj && (n += 1)
#             end

#             clust = zeros(Int, n)
#             m = 1
#             for (j, lj) in enumerate(labels)
#                 li == lj && (clust[m] = j)
#                 m += 1
#             end
#         end
#     end
#     return clusters
# end

# function DBSCAN(points, radius, min_pts; metric = Euclidean(), query_func = )

# end

# """ expermiental
# sorts the points in place
# """
# function hl_dbscan_exp!(points::Vector{SVector{6, T}}, radius, min_pts; metric = Euclidean(), n_threads = 1, bin_size = 1_000, r_stop = 10) where {T}
#     # # choose a resolution given the binsize estimate
#     # n_bins = length(points) / bin_size
#     # d      = 2^ceil(Int, log2(n_bins/12)/2)
#     # res    = Resolution(d)
#     # r_pix  = sqrt(4pi / res.numOfPixels)
#     # # check against the requested clustering radius
#     # # we want the bins to be large relative to the distance between points in a cluster
#     # if r_pix < 10radius
#         # 
#     d   = 1
#     res = Resolution(d)
#     while length(points) / res.numOfPixels > bin_size && sqrt(4pi / res.numOfPixels) > r_stop*radius
#         d   *= 2
#         res  = Resolution(d)
#     end
#     r_pix = sqrt(4pi / res.numOfPixels)

#     # point to pixel map and pixel start, stop index maps
#     r2 = radius^2
#     reverse_map  = collect(eachindex(points))  # need for querying tracklet info
#     point_to_pix = zeros(Int, length(points))

#     Threads.@threads for (i, p) in pairs(points)
#         # convert position to angles
#         a = sign(p[2])*acos(p[1]/sqrt(p[1]^2 + p[2]^2))
#         b = pi/2 - acos(p[3]/sqrt(p[1]^2 + p[2]^2 + p[3]^2))
#         # get pixel
#         point_to_pix[i] = ang2pixRing(res, a, b)
#     end

#     # sort so that points in a pixel are stored consecutively
#     perm         = sortperm(point_to_pix)
#     point_to_pix = point_to_pix[perm]
#     points       = points[perm]
#     reverse_map  = reverse_map[perm]

#     # do a set up pass to accelerate queries (look up range for each pixel)
#     pixel_ranges = [0:0 for _ in 1:res.numOfPixels]
#     current_pix = 1
#     start_idx   = 1
#     for i in eachindex(point_to_pix)
#         if point_to_pix[i] != current_pix
#             pixel_ranges[current_pix] = start_idx:i-1
#             current_pix = point_to_pix[i]
#             start_idx = i
#         end
#     end

#     # now for the clustering
#     # for each cell, we'll query a disk around it's center to get the neighboring cells
#     # check each point against every other in the cell and surrounding cells
#     mergers = [Tuple{Int, Int} for _ in 1:n_threads]
#     Threads.@threads for i_t in 1:n_threads
#         for i in i_t:n_threads:res.numOfPixels
#             a, b = pix2angRing(res, i)
#             neighbor_pixels = queryDiscRing(res, a, b, 2r_pix)


#         end
#     end

#     # clean up sweep for merger flags

#     # 
# end

# function healpix_neighbors_ring(i, res)
#     neighbors = zeros(Int, 8)

# end

"""
Experimental DBSCAN

uses a cartesian binning to assign points to bins with width of the clustering radius. A hash map is built to point to an arbitray
"first" point in each bin, and a (linked) list is built to point each each to the next in the bin. Then each bin is compared
to its neighbors to find clusters in the typical dbscan fashion.

currently uses the unweighted cartesian norm as the distance metric, since the bin size is metric dependent

This should be better than kd trees for low dimensionsal data. For high dimensional its unclear to me how well this will perform,
but kdtrees aren't much good in that case anyways.

This is perhaps not technically dbscan, as the neighborhood count is done after the clustering (and so may yield different clusters),
but it's a lot simpler and should be sufficient for heliolinc

This is also a general purpose implementation, in the sense that binning is not dependent on domain specific dsitributions
or prior information about the dataset, in contrast to the implementation using healpix, which is heliolinc specific.
"""
function dbscan_cart_hash(points::AbstractVector{SVector{D, T}}, radius::T, min_pts::Int; n_threads = 1) where {D, T}
    # instantiate cell look up
    # points each cell to the first point in the linked list in the cell 
    cells  = Dict{SVector{D, Int}, Int}()
    center = mean(points)

    # how to parallelize this part?
    links = zeros(Int, length(points))
    for (i, p) in pairs(points)
        # compute the cell 
        cell = find_cell(p - center, radius)
        if haskey(cells, cell)
            # link point to next point in cell
            j = cells[cell]
            links[i] = j
        end
        cells[cell] = i
    end

    # we can reorder the points now to get better cache locality during the comparison
    # but does it improve the speed enought to offset the cost of the sort?
    # probably, sorting is pretty fast

    # run typical dbscan kernel by comparing points between neighboring cells
    # distribute cells across threads (by using larger chunks of cells)
    # currently using 64-bit UInt for cluster label, but that may be excessive

    chunks = collect(Iterators.partition(sort(collect(keys(cells))), n_threads))
    merges = [Tuple{Int, Int}[] for _ in chunks]
    labels = zeros(Int, length(points))
    Threads.@threads for i_c in 1:n_threads
        chunk = chunks[i_c]
        merge = merges[i_c]
        for cell in chunk
            # run dbscan for self
            dbscan_kernel_cart_hash!(labels, cells[cell], cells[cell], links, points, radius)
            # look up neighbors
            # only check the forward neighbors, since backward neighbors
            # will check this cell as a forward neighbor
            for d in 1:D
                neighbor = MVector{D}(cell)
                neighbor[d] += 1
                neighbor = SVector{D}(neighbor)
                # if neighbor is in chunk
                if !haskey(cells, neighbor)
                    continue
                end
                if in(neighbor, chunk)
                    dbscan_kernel_cart_hash!(labels, cells[cell], cells[neighbor], links, points, radius)
                else
                    dbscan_flag_merges_cart_hash!(merge, cells[cell], cells[neighbor], links, points, radius)
                end
            end
        end
    end

    # clean up the merges
    for (i, j) in Iterators.flatten(merges)
        dbscan.join_labels!(labels, i, j)
    end

    # check each cluster for min pts
    cluster_counts = Dict{UInt, Int}()
    for l in labels
        if haskey(cluster_counts, l)
            cluster_counts[l] += 1
        else
            cluster_counts[l] = 1
        end
    end

    for (i, l) in pairs(labels)
        if cluster_counts[l] < min_pts
            labels[i] = 0
        end
    end

    return labels
end

function dbscan_kernel_cart_hash!(labels, i, j, links, points, radius)
    ii = i
    while ii != 0
        jj = j
        while jj != 0 
            # check the two points
            if ii < jj && dot(points[ii] - points[jj], points[ii] - points[jj]) <= radius^2
                dbscan.join_labels!(labels, li, lj)
            end
            jj = links[jj]
        end
        ii = links[ii]
    end
end

function dbscan_flag_merges_cart_hash!(merges, i, j, links, points, radius)
    ii = i
    while ii != 0
        # first count the number of neighbors
        jj = j
        while jj != 0 
            # check the two points
            if ii < jj && dot(points[ii] - points[jj], points[ii] - points[jj]) <= radius^2
                push!(merges, (i, j))
            end
            jj = links[jj]
        end
        ii = links[ii]
    end
end

function find_cell(point::SVector{D, T}, radius::T) where {D, T}
    cell = @MVector zeros(Int, D)
    for d in eachindex(point)
        cell[d] = sign(point[d]) * floor(Int, abs(point[d] / radius)) # div return only the integer component
    end
    return SVector{D}(cell)
end
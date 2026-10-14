function cluster_all(points, radius, min_pts; n_threads = 1, max_pts = Inf)
    tree     = KDTree(points)
    clusters = inrange(tree, points, radius)

    filter!(x -> length(x) >= min_pts, clusters)

    return clusters
end

function cluster_unique(points, radius, min_pts; n_threads = 1, max_pts = Inf)
    # use a kdtree lookup to assign labels
    # if there is a collision, accept only the largest cluster (have to track cluster sizes)
    tree   = KDTree(points)
    labels = zeros(Int, length(points))
    counts = zeros(Int, length(points))
    for thr in 1:n_threads
        neighbors = Int[]
        for i in thr:n_threads:length(points)
            if labels[i] != 0
                continue
            end

            empty!(neighbors)
            inrange!(neighbors, tree, points[i], radius)
            if length(neighbors) >= min_pts
                for j in neighbors
                    labels[j] = i
                    counts[j] = length(neighbors)
                end
            end
        end
    end
    return labels
end

function cluster_cells_old(points::AbstractVector{SVector{D, T}}, radius, min_pts; n_threads = 1, max_pts = Inf) where {D, T}
    cells = Dict{SVector{D, Int32}, Vector{Int}}()
    center = mean(points)

    # assign points to cells
    for (i, p) in pairs(points)
        cell = SVector{D, Int32}(find_cell(p - center, radius))
        if haskey(cells, cell)
            push!(cells[cell], i)
        else
            cells[cell] = [i]
        end
    end

    # skipped overfilled cells
    n_deleted = 0
    for (cell, idx) in cells
        if length(idx) > max_pts
            delete!(cells, cell)
            n_deleted += 1
        end
    end

    _neighbor_cells = map(x -> SVector{D}(x), Iterators.product(((0, 1) for _ in 1:D)...))

    # domain decomposition
    n_leafs  = 1
    leafsize = length(points)
    while n_leafs < n_threads
        n_leafs *= 2
        leafsize = leafsize - leafsize ÷ 2
    end
    tree   = KDTree(points; leafsize = leafsize, reorder = false)
    chunks = [NearestNeighbors.get_leaf_range(tree.tree_data, i+tree.tree_data.n_internal_nodes) for i in 1:tree.tree_data.n_leafs]
    chunk_keys = collect(keys(chunks))

    # TODO: flag points, if enough in neighborhood are already flagged, skip
    # assigned = [false for p in points]

    idx_map = zeros(UInt, length(points))
    for (i, j) in enumerate(tree.indices)
        idx_map[j] = i
    end

    # clusters = [Vector{Vector{Int}}(undef, 0) for _ in 1:n_threads]
    labels = [zeros(UInt, length(points)) for _ in chunks]
    Threads.@threads for i_th in 1:n_threads
        for i_c in i_th:n_threads:length(chunks)
            chunk = chunks[chunk_keys[i_c]]
            cluster = Int[]
            for i_tree in chunk
                i = idx_map[i_tree]
                p_i = points[i]
                celli = find_cell(p_i - center, radius)
                n_core = 0
                for _n in _neighbor_cells
                    cellj = celli + _n

                    if !haskey(cells, cellj)
                        continue
                    end

                    for j in cells[cellj]
                        p_j = points[j]
                        if dot(p_i - p_j, p_i - p_j) < radius^2
                            n_core += 1
                        end
                    end
                end

                if n_core >= min_pts # is core point
                    for _n in _neighbor_cells
                        cellj = celli + _n

                        if !haskey(cells, cellj)
                            continue
                        end

                        for j in cells[cellj]

                            # collect the cluster
                            p_j = points[j]
                            if dot(p_i - p_j, p_i - p_j) < radius^2
                                # join_labels!(labels[i_c], i, j)
                                # push!(cluster, j)
                                # labels[i_c]
                                labels[i_c][j] = i
                            end
                        end
                    end
                end
                # push!(clusters[i_c], cluster)
            end
        end
    end

    final_labels = zeros(UInt, length(points))
    for lchunk in labels
        for (i, l) in pairs(lchunk)
            if l != 0
                final_labels[i] = l
            end
        end
    end
    return final_labels
    # return Iterators.flatten(clusters)
end

function find_cell(point::SVector{D, T}, radius) where {D, T}
    cell = @MVector zeros(Int, D)
    for d in eachindex(point)
        cell[d] = sign(point[d]) * floor(Int, abs(point[d] / radius)) 
    end
    return SVector{D}(cell)
    # return join(cell, ',')
end

# function bin_clusters(points::Vector{SVector{D, T}}, radius, min_pts; n_threads = 1, max_pts = 1_000) where {D, T}    
#     center = mean(points)

#     bins = Dict{
#         SVector{D, Int}
#     }()
#     for p in points

#     end

#     # labels = Vector{SVector{D, Int}}(undef, length(points))
#     # Threads.@threads for p in points
#     #     # compute the cell index for each point
#     # end

#     # cells = unique(labels)
#     # bins  = Dict(cell => Int for cell in cells)
#     # Threads.@threads for cell in cells
#     #     # collect points into bins
#     #     for (i, l) in enumerate(labels)
#     #         if l == cell
#     #             push!(bins[l], i)
#     #         end
#     #     end
#     # end

#     # # find all existing corners
#     # corners = Dict(k => true for k in keys(bins))
#     # for k in keys(bins)
#     #     for n in neighbors(k)
#     #         # adds key if necessary
#     #         corners[n] = true
#     #     end
#     # end

#     # # for each corner, dump a cluster (will likely duplicate, but thats the price of this method)
#     # clusters = [Vector{Vector{Int}}(undef, 0) for _ in 1:n_threads]
#     # Threads.@threads for thread in 1:n_threads
#     #     for i in thread:n_threads:length(corners)
#     #         push!(clusters[threads], [])
#     #         # get cells
#     #         corner = corners[i]
#     #         for offset in 
#     #             cell = corner + offset
#     #             push!(clusters[threads])
#     #         end
#     #     end
#     # end

#     return clusters
# end

# function neighbors(cell)

# end
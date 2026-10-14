"""
indices maps the points in the partition back to the original array
"""
struct hp_partition{P}
    points ::Vector{P}
    indices::Vector{Int}
end

function hp_partition_arrows(points::Vector{SVector{6, T}}, res; n_threads = 1) where {T}
    points_to_pixels = zeros(Int, length(points))

    # assign points to chunks
    Threads.@threads for i_t in 1:n_threads
        for i in i_t:n_threads:length(points)
            pos  = SVector{3}(view(points[i], 1:3))
            a, b = vec2ang(normalize(pos)...)
            points_to_pixels[i] = ang2pixRing(res, a, b)
        end
    end

    # collect points to chunks
    chunks = Vector{hp_partition{SVector{6, T}}}(undef, res.numOfPixels)
    Threads.@threads for i_t in 1:n_threads
        for i_p in i_t:n_threads:res.numOfPixels
            n = 0
            for l in points_to_pixels
                if l == i_p
                    n += 1
                end
            end

            chunk_points = similar(points, n)
            chunk_ids    = zeros(Int, n)
            m = 1
            for (i, l) in pairs(points_to_pixels)
                if l == i_p
                    chunk_points[m] = points[i]
                    chunk_ids[m]    = i
                    m += 1
                end
            end

            chunks[i_p] = hp_partition(chunk_points, chunk_ids)
        end
    end

    return chunks, points_to_pixels
end
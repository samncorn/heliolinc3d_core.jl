# I3 = SMatrix{3, 3, Float64}([
#     1.0 0.0 0.0;
#     0.0 1.0 0.0;
#     0.0 0.0 1.0;
# ])

""" Convert from spherical to cartesian coordinates

Returns a tuple (x, y, z)
"""
function sphere2cart(long, lat; r=1.0, deg=false)
    if deg
        return (
            r*cosd(lat)*cosd(long),
            r*cosd(lat)*sind(long),
            r*sind(lat),
        )
    else
        return (
            r*cos(lat)*cos(long),
            r*cos(lat)*sin(long),
            r*sin(lat),
        )
    end
end

function cart2sphere(x, y, z)
    a = atan(y, x)
    b = atan(z, sqrt(x^2 + y^2))

    if a < 0
        a += 2pi
    end

    return a, b
end

""" 
calculate the intersection distance of a line with a sphere.

There are 0, 1 or 2 solutions. In order to maintain a type stable output, a tuple of 2 solutions is always returned,
with NaNs for invalid results

arguments
=========
o
    vector from origin to a point on the line
l
    vector parameterizing the line
r
    radius of the sphere
"""
function sphere_intersect(o, l, r)
    # normalize l
    l_hat = l ./ norm(l)
    ol = dot(o, l_hat)
    o2 = sum(dot(o, o))

    discrim = ol^2 - o2 + r^2
    if discrim >= 0 
        k1 = -ol + sqrt(discrim)
        k2 = -ol - sqrt(discrim)
        return (k1, k2)
    else
        return (NaN*discrim, NaN*discrim)
    end
end

""" manually specify which branch to take
"""
function sphere_intersect_branch(o, l, r; neg_branch :: Bool = false)
    # normalize l
    l_hat = l ./ norm(l)
    ol = dot(o, l_hat)
    o2 = sum(dot(o, o))

    discrim = ol^2 - o2 + r^2
    discrim_root = discrim < 0 ? NaN : sqrt(discrim) 
    # return 
    if neg_branch
        return -ol - discrim_root
    else
        return -ol + discrim_root
    end
end

""" 
May be unstable for nearly antipodal points
returns same units (radians or degrees) as input.
"""
function haversine(long1, long2, lat1, lat2; deg=false, r=1.0)
    if deg 
        return _haversined(long1, long2, lat1, lat2; r=r) 
    else
        return _haversine(long1, long2, lat1, lat2; r=r) 
    end
end

function _haversine(long1, long2, lat1, lat2; r=1.0)
    dlong = long2 - long1
    dlat = lat2 - lat1
    return 2r*asin(sqrt(0.5*(
        1.0 - cos(dlat) + cos(lat1)*cos(lat2)*(1.0 - cos(dlong)) 
)))
end

function _haversined(long1, long2, lat1, lat2; r=1.0)
    dlong = long2 - long1
    dlat = lat2 - lat1
    return 2r*asind(sqrt(0.5*(
        1.0 - cosd(dlat) + cosd(lat1)*cosd(lat2)*(1.0 - cosd(dlong)) 
)))
end

finiteDiff(x1, x2, t1, t2) = (x2-x1) / (t2-t1)

# function rotateX(x)

# end

function rotateY(x)
    return SMatrix{3, 3}((
        cos(x), 0.0, -sin(x), 
           0.0, 1.0,     0.0, 
        sin(x), 0.0,  cos(x)
    ))
end

function rotateZ(x)
    SMatrix{3, 3}((
         cos(x), sin(x), 0.0,
        -sin(x), cos(x), 0.0,
            0.0,    0.0, 1.0
))
end

# NEVERMIND NOT CORRECT
# function tracklet_inclination(ang1, ang2; R_ecl = I3)
#     x, y, z = R_ecl*(ang1 - ang2)
#     return abs(atan(z, sqrt(x^2 + y^2)))
# end
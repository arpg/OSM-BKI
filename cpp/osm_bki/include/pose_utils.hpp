#pragma once

#include <cmath>
#include <vector>

#include "continuous_bki.hpp"
#include "file_io.hpp"

namespace continuous_bki {

// -- Row-major 4x4 helpers (no Eigen dependency) -------------------------

inline Transform4x4 mat4_multiply(const Transform4x4& A, const Transform4x4& B) {
    Transform4x4 C;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k)
                s += A.m[i * 4 + k] * B.m[k * 4 + j];
            C.m[i * 4 + j] = s;
        }
    return C;
}

// Inverse of a rigid-body transform: inv([R t; 0 1]) = [R^T  -R^T*t; 0  1]
inline Transform4x4 mat4_rigid_inverse(const Transform4x4& T) {
    Transform4x4 result;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.m[i * 4 + j] = T.m[j * 4 + i];
    for (int i = 0; i < 3; ++i) {
        double s = 0.0;
        for (int k = 0; k < 3; ++k)
            s += result.m[i * 4 + k] * T.m[k * 4 + 3];
        result.m[i * 4 + 3] = -s;
    }
    result.m[12] = 0.0;
    result.m[13] = 0.0;
    result.m[14] = 0.0;
    result.m[15] = 1.0;
    return result;
}

// -- Quaternion to row-major 4x4 -----------------------------------------
//
// Normalises the quaternion before building the matrix.
// Argument order: (qx, qy, qz, qw) -- same as scipy / ROS convention.

inline Transform4x4 poseToTransform(double px, double py, double pz,
                                    double qx, double qy, double qz, double qw) {
    double n = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if (n > 0.0) { qx /= n; qy /= n; qz /= n; qw /= n; }

    Transform4x4 T;
    T.m[0]  = 1.0 - 2.0*(qy*qy + qz*qz);
    T.m[1]  = 2.0*(qx*qy - qz*qw);
    T.m[2]  = 2.0*(qx*qz + qy*qw);
    T.m[3]  = px;
    T.m[4]  = 2.0*(qx*qy + qz*qw);
    T.m[5]  = 1.0 - 2.0*(qx*qx + qz*qz);
    T.m[6]  = 2.0*(qy*qz - qx*qw);
    T.m[7]  = py;
    T.m[8]  = 2.0*(qx*qz - qy*qw);
    T.m[9]  = 2.0*(qy*qz + qx*qw);
    T.m[10] = 1.0 - 2.0*(qx*qx + qy*qy);
    T.m[11] = pz;
    T.m[12] = 0.0;
    T.m[13] = 0.0;
    T.m[14] = 0.0;
    T.m[15] = 1.0;
    return T;
}

// -- High-level transform builders ----------------------------------------
//
// Build the lidar-to-map (world) transform matching visualize_map_osm.cpp:
//   body_to_world    = poseToTransform(pose)
//   body_to_world.t -= init_rel_pos
//   lidar_to_map     = body_to_world * inv(body_to_lidar)

inline Transform4x4 buildLidarToMap(const PoseRecord& pose,
                                    const Transform4x4& body_to_lidar,
                                    const double* init_rel_pos = nullptr) {
    Transform4x4 body_to_world = poseToTransform(
        pose.x, pose.y, pose.z,
        pose.qx, pose.qy, pose.qz, pose.qw);
    if (init_rel_pos) {
        body_to_world.m[3]  -= init_rel_pos[0];
        body_to_world.m[7]  -= init_rel_pos[1];
        body_to_world.m[11] -= init_rel_pos[2];
    }
    return mat4_multiply(body_to_world, mat4_rigid_inverse(body_to_lidar));
}

// -- Per-point transforms -------------------------------------------------

inline Point3D transformPoint(const Transform4x4& T, const Point3D& p) {
    double x = static_cast<double>(p.x);
    double y = static_cast<double>(p.y);
    double z = static_cast<double>(p.z);
    return Point3D(
        static_cast<float>(T.m[0]*x  + T.m[1]*y  + T.m[2]*z  + T.m[3]),
        static_cast<float>(T.m[4]*x  + T.m[5]*y  + T.m[6]*z  + T.m[7]),
        static_cast<float>(T.m[8]*x  + T.m[9]*y  + T.m[10]*z + T.m[11]));
}

inline void transformPointsInPlace(const Transform4x4& T,
                                   std::vector<Point3D>& points) {
    for (auto& p : points)
        p = transformPoint(T, p);
}

// -- Convenience: build matrix + transform in one call --------------------

inline std::vector<Point3D> transformScanToWorld(
    const std::vector<Point3D>& points,
    double px, double py, double pz,
    double qx, double qy, double qz, double qw,
    const Transform4x4& body_to_lidar,
    const double* init_rel_pos = nullptr)
{
    PoseRecord pose;
    pose.x = px;  pose.y = py;  pose.z = pz;
    pose.qx = qx; pose.qy = qy; pose.qz = qz; pose.qw = qw;
    Transform4x4 lidar_to_map = buildLidarToMap(pose, body_to_lidar, init_rel_pos);
    std::vector<Point3D> out(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        out[i] = transformPoint(lidar_to_map, points[i]);
    return out;
}

}  // namespace continuous_bki

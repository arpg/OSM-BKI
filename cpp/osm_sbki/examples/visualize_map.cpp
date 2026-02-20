#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <array>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "dataset_utils.hpp"
#include "file_io.hpp"

namespace {

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

Color colorFromLabel(uint32_t label) {
    // Deterministic pseudo-palette from label ID.
    const uint32_t h = label * 2654435761u;
    return Color{
        static_cast<uint8_t>((h >> 16) & 0xFF),
        static_cast<uint8_t>((h >> 8) & 0xFF),
        static_cast<uint8_t>(h & 0xFF),
    };
}

void normalizeQuaternion(double& qx, double& qy, double& qz, double& qw) {
    const double n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (n > 1e-12) {
        qx /= n;
        qy /= n;
        qz /= n;
        qw /= n;
    } else {
        qx = 0.0;
        qy = 0.0;
        qz = 0.0;
        qw = 1.0;
    }
}

continuous_bki::Transform4x4 poseToMatrix(const continuous_bki::PoseRecord& pose) {
    double qx = pose.qx;
    double qy = pose.qy;
    double qz = pose.qz;
    double qw = pose.qw;
    normalizeQuaternion(qx, qy, qz, qw);

    const double xx = qx * qx;
    const double yy = qy * qy;
    const double zz = qz * qz;
    const double xy = qx * qy;
    const double xz = qx * qz;
    const double yz = qy * qz;
    const double wx = qw * qx;
    const double wy = qw * qy;
    const double wz = qw * qz;

    continuous_bki::Transform4x4 t;
    t.m = {
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),       pose.x,
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),       pose.y,
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy), pose.z,
        0.0,                   0.0,                   0.0,                   1.0
    };
    return t;
}

continuous_bki::Transform4x4 multiply(
    const continuous_bki::Transform4x4& a,
    const continuous_bki::Transform4x4& b) {
    continuous_bki::Transform4x4 c;
    for (int r = 0; r < 4; ++r) {
        for (int col = 0; col < 4; ++col) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) {
                v += a.m[static_cast<size_t>(r * 4 + k)] * b.m[static_cast<size_t>(k * 4 + col)];
            }
            c.m[static_cast<size_t>(r * 4 + col)] = v;
        }
    }
    return c;
}

continuous_bki::Transform4x4 inverseRigid(const continuous_bki::Transform4x4& t) {
    // Inverse for rigid transform [R t; 0 1] is [R^T -R^T t; 0 1].
    continuous_bki::Transform4x4 inv;
    const double r00 = t.m[0],  r01 = t.m[1],  r02 = t.m[2],  tx = t.m[3];
    const double r10 = t.m[4],  r11 = t.m[5],  r12 = t.m[6],  ty = t.m[7];
    const double r20 = t.m[8],  r21 = t.m[9],  r22 = t.m[10], tz = t.m[11];

    inv.m[0] = r00; inv.m[1] = r10; inv.m[2] = r20;
    inv.m[4] = r01; inv.m[5] = r11; inv.m[6] = r21;
    inv.m[8] = r02; inv.m[9] = r12; inv.m[10] = r22;

    inv.m[3] = -(inv.m[0] * tx + inv.m[1] * ty + inv.m[2] * tz);
    inv.m[7] = -(inv.m[4] * tx + inv.m[5] * ty + inv.m[6] * tz);
    inv.m[11] = -(inv.m[8] * tx + inv.m[9] * ty + inv.m[10] * tz);

    inv.m[12] = 0.0; inv.m[13] = 0.0; inv.m[14] = 0.0; inv.m[15] = 1.0;
    return inv;
}

continuous_bki::Point3D transformPoint(
    const continuous_bki::Transform4x4& t,
    const continuous_bki::Point3D& p) {
    const double x = p.x;
    const double y = p.y;
    const double z = p.z;
    return continuous_bki::Point3D(
        static_cast<float>(t.m[0] * x + t.m[1] * y + t.m[2] * z + t.m[3]),
        static_cast<float>(t.m[4] * x + t.m[5] * y + t.m[6] * z + t.m[7]),
        static_cast<float>(t.m[8] * x + t.m[9] * y + t.m[10] * z + t.m[11]));
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config_path = (argc > 1) ? argv[1] : "configs/mcd_config.yaml";

    continuous_bki::DatasetConfig config;
    std::string error_msg;
    if (!continuous_bki::loadDatasetConfig(config_path, config, error_msg)) {
        std::cerr << "Config error: " << error_msg << std::endl;
        return 1;
    }
    const size_t skip_frames = (argc > 2)
        ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10))
        : static_cast<size_t>(config.skip_frames);
    const size_t step = skip_frames + 1;

    std::vector<continuous_bki::ScanLabelPair> pairs;
    if (!continuous_bki::collectScanLabelPairs(config, pairs, error_msg)) {
        std::cerr << "Dataset error: " << error_msg << std::endl;
        return 1;
    }

    std::vector<continuous_bki::PoseRecord> poses;
    if (!continuous_bki::readPosesCSV(config.pose_path, poses)) {
        std::cerr << "Failed to read poses: " << config.pose_path << std::endl;
        return 1;
    }
    continuous_bki::Transform4x4 body_to_lidar;
    if (!continuous_bki::readBodyToLidarCalibration(config.calibration_path, body_to_lidar)) {
        std::cerr << "Failed to read calibration: " << config.calibration_path << std::endl;
        return 1;
    }
    const continuous_bki::Transform4x4 lidar_to_body = inverseRigid(body_to_lidar);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->points.clear();
    cloud->is_dense = false;

    std::unordered_map<int, continuous_bki::Transform4x4> lidar_to_map_by_scan_id;
    lidar_to_map_by_scan_id.reserve(poses.size());
    if (poses.empty()) {
        std::cerr << "No poses loaded from " << config.pose_path << std::endl;
        return 1;
    }

    // Match mcd_util behavior: make poses relative to first pose.
    const continuous_bki::Transform4x4 first_body_to_world = poseToMatrix(poses.front());
    const continuous_bki::Transform4x4 world_to_first = inverseRigid(first_body_to_world);

    for (const auto& pose : poses) {
        const continuous_bki::Transform4x4 body_to_world = poseToMatrix(pose);
        const continuous_bki::Transform4x4 body_to_world_rel = multiply(world_to_first, body_to_world);
        const continuous_bki::Transform4x4 lidar_to_map = multiply(body_to_world_rel, lidar_to_body);
        lidar_to_map_by_scan_id[pose.scan_id] = lidar_to_map;
    }

    const size_t scans_to_use = pairs.size();
    size_t scans_loaded = 0;
    size_t scans_skipped = 0;

    for (size_t s = 0; s < scans_to_use; s += step) {
        const auto& pair = pairs[s];

        int scan_id = -1;
        try {
            scan_id = std::stoi(pair.scan_id);
        } catch (...) {
            ++scans_skipped;
            continue;
        }

        auto pose_it = lidar_to_map_by_scan_id.find(scan_id);
        if (pose_it == lidar_to_map_by_scan_id.end()) {
            ++scans_skipped;
            continue;
        }

        std::vector<continuous_bki::Point3D> points;
        std::vector<uint32_t> labels;
        if (!continuous_bki::readPointCloudBin(pair.scan_path, points) ||
            !continuous_bki::readLabelBin(pair.label_path, labels) ||
            points.size() != labels.size()) {
            ++scans_skipped;
            continue;
        }

        cloud->points.reserve(cloud->points.size() + points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const continuous_bki::Point3D world_pt = transformPoint(pose_it->second, points[i]);
            const Color c = colorFromLabel(labels[i]);

            pcl::PointXYZRGB p;
            p.x = world_pt.x;
            p.y = world_pt.y;
            p.z = world_pt.z;
            p.r = c.r;
            p.g = c.g;
            p.b = c.b;
            cloud->points.push_back(p);
        }
        ++scans_loaded;
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;

    if (cloud->points.empty()) {
        std::cerr << "No points accumulated. Loaded scans: " << scans_loaded
                  << ", skipped scans: " << scans_skipped << std::endl;
        return 1;
    }

    std::cout << "Visualizing accumulated cloud using skip_frames=" << skip_frames
              << " (step=" << step << ") over " << scans_to_use
              << " pairs. Loaded scans: " << scans_loaded
              << ", skipped scans: " << scans_skipped
              << ", total points: " << cloud->points.size() << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("MCD Accumulated Viewer"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "scan_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scan_cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spin();
    }

    return 0;
}

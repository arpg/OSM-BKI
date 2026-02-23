#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include "ankerl/unordered_dense.h"
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "dataset_utils.hpp"
#include "file_io.hpp"
#include "osm_parser.hpp"

namespace {

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

Color colorFromLabel(uint32_t label) {
    const uint32_t h = label * 2654435761u;
    return Color{
        static_cast<uint8_t>((h >> 16) & 0xFF),
        static_cast<uint8_t>((h >> 8) & 0xFF),
        static_cast<uint8_t>(h & 0xFF),
    };
}

Color colorForOSM(const osm_parser::Polyline2D& way) {
    switch (way.category) {
        case osm_parser::Category::Building: return {30, 180, 30};
        case osm_parser::Category::Road: return {240, 120, 20};
        case osm_parser::Category::Sidewalk: return {220, 220, 220};
        case osm_parser::Category::Parking: return {245, 210, 80};
        case osm_parser::Category::Fence: return {170, 120, 70};
        case osm_parser::Category::Stairs: return {150, 100, 60};
        case osm_parser::Category::Grassland: return {60, 170, 80};
        case osm_parser::Category::Tree: return {20, 130, 20};
        case osm_parser::Category::Unknown: break;
    }
    return {140, 140, 140};
}

Eigen::Matrix4d poseToMatrix(const continuous_bki::PoseRecord& pose) {
    Eigen::Quaterniond quat(pose.qw, pose.qx, pose.qy, pose.qz);
    quat.normalize();
    Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
    t.block<3, 3>(0, 0) = quat.toRotationMatrix();
    t(0, 3) = pose.x;
    t(1, 3) = pose.y;
    t(2, 3) = pose.z;
    return t;
}

continuous_bki::Point3D transformPoint(const Eigen::Matrix4d& t, const continuous_bki::Point3D& p) {
    const Eigen::Vector4d v(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z), 1.0);
    const Eigen::Vector4d out = t * v;
    return continuous_bki::Point3D(
        static_cast<float>(out(0)),
        static_cast<float>(out(1)),
        static_cast<float>(out(2)));
}

std::pair<double, double> transformOSMPointLikeBKI(
    double x,
    double y) {
    // OSM parser already outputs local ENU coordinates around osm_origin_lat/lon.
    // With kth_day_06 as anchor, keep OSM in that local anchor frame.
    return {x, y};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mcd_config_path =
        (argc > 1) ? argv[1] : "configs/mcd_config.yaml";
    std::string osm_config_path;
    if (argc > 2) {
        osm_config_path = argv[2];
    } else {
        // Default: examples/osm_config.yaml next to this executable (build/../examples/)
        const auto exe_dir = std::filesystem::path(argc > 0 ? argv[0] : "").parent_path();
        osm_config_path = (exe_dir / ".." / "examples" / "osm_config.yaml").lexically_normal().string();
    }

    continuous_bki::DatasetConfig map_config;
    std::string error_msg;
    if (!continuous_bki::loadDatasetConfig(mcd_config_path, map_config, error_msg)) {
        std::cerr << "Map config error: " << error_msg << std::endl;
        return 1;
    }
    const size_t skip_frames = (argc > 3)
        ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10))
        : static_cast<size_t>(map_config.skip_frames);
    const size_t step = skip_frames + 1;

    std::vector<continuous_bki::ScanLabelPair> pairs;
    if (!continuous_bki::collectScanLabelPairs(map_config, pairs, error_msg)) {
        std::cerr << "Dataset error: " << error_msg << std::endl;
        return 1;
    }

    std::vector<continuous_bki::PoseRecord> poses;
    if (!continuous_bki::readPosesCSV(map_config.pose_path, poses)) {
        std::cerr << "Failed to read poses: " << map_config.pose_path << std::endl;
        return 1;
    }
    if (poses.empty()) {
        std::cerr << "No poses loaded from " << map_config.pose_path << std::endl;
        return 1;
    }

    continuous_bki::Transform4x4 body_to_lidar;
    if (!continuous_bki::readBodyToLidarCalibration(map_config.calibration_path, body_to_lidar)) {
        std::cerr << "Failed to read calibration: " << map_config.calibration_path << std::endl;
        return 1;
    }
    const Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> body_to_lidar_mat(body_to_lidar.m.data());
    const Eigen::Matrix4d lidar_to_body = body_to_lidar_mat.inverse();

    Eigen::Vector3d init_rel_pos(0.0, 0.0, 0.0);
    if (map_config.use_init_rel_pos) {
        init_rel_pos(0) = map_config.init_rel_pos_x;
        init_rel_pos(1) = map_config.init_rel_pos_y;
        init_rel_pos(2) = map_config.init_rel_pos_z;
    }

    ankerl::unordered_dense::map<int, Eigen::Matrix4d> lidar_to_map_by_scan_id;
    lidar_to_map_by_scan_id.reserve(poses.size());

    for (const auto& pose : poses) {
        Eigen::Matrix4d body_to_world_rel = poseToMatrix(pose);
        body_to_world_rel(0, 3) -= init_rel_pos(0);
        body_to_world_rel(1, 3) -= init_rel_pos(1);
        body_to_world_rel(2, 3) -= init_rel_pos(2);
        Eigen::Matrix4d lidar_to_map = body_to_world_rel * lidar_to_body;

        lidar_to_map_by_scan_id[pose.scan_id] = lidar_to_map;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    map_cloud->points.clear();
    map_cloud->is_dense = false;

    size_t scans_loaded = 0;
    size_t scans_skipped = 0;
    for (size_t s = 0; s < pairs.size(); s += step) {
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

        map_cloud->points.reserve(map_cloud->points.size() + points.size());
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
            map_cloud->points.push_back(p);
        }
        ++scans_loaded;
    }

    if (map_cloud->points.empty()) {
        std::cerr << "No map points accumulated. Loaded scans: " << scans_loaded
                  << ", skipped scans: " << scans_skipped << std::endl;
        return 1;
    }
    map_cloud->width = static_cast<uint32_t>(map_cloud->points.size());
    map_cloud->height = 1;

    osm_parser::OSMConfig osm_config;
    if (!osm_parser::loadOSMConfig(osm_config_path, osm_config, error_msg)) {
        std::cerr << "OSM config error: " << error_msg << std::endl;
        return 1;
    }
    // Resolve OSM path: relative to dataset_root_path (legacy) or config file directory
    if (!map_config.osm_file.empty()) {
        if (!map_config.dataset_root_path.empty()) {
            osm_config.osm_file = (std::filesystem::path(map_config.dataset_root_path) / map_config.osm_file).string();
        } else {
            const auto config_dir = std::filesystem::path(mcd_config_path).parent_path();
            osm_config.osm_file = (config_dir / map_config.osm_file).lexically_normal().string();
        }
    }
    if (map_config.use_osm_origin_from_mcd) {
        osm_config.use_origin_override = true;
        osm_config.osm_origin_lat = map_config.osm_origin_lat;
        osm_config.osm_origin_lon = map_config.osm_origin_lon;
    }
    if (osm_config.osm_file.empty()) {
        std::cerr << "No OSM file configured: set osm_file in " << mcd_config_path << " or in " << osm_config_path << std::endl;
        return 1;
    }
    osm_parser::ParsedOSMData osm_data;
    if (!osm_parser::parsePolylines(osm_config, osm_data, error_msg)) {
        std::cerr << "OSM parse error: " << error_msg << std::endl;
        return 1;
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Map + OSM Viewer"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);
    viewer->addPointCloud<pcl::PointXYZRGB>(map_cloud, "map_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "map_cloud");

    int segment_count = 0;
    for (size_t i = 0; i < osm_data.polylines.size(); ++i) {
        const auto& way = osm_data.polylines[i];
        if (way.points.size() < 2) continue;
        const Color c = colorForOSM(way);
        for (size_t j = 0; j + 1 < way.points.size(); ++j) {
            const auto a = transformOSMPointLikeBKI(way.points[j].first, way.points[j].second);
            const auto b = transformOSMPointLikeBKI(way.points[j + 1].first, way.points[j + 1].second);
            pcl::PointXYZ pa(static_cast<float>(a.first), static_cast<float>(a.second), 0.05f);
            pcl::PointXYZ pb(static_cast<float>(b.first), static_cast<float>(b.second), 0.05f);
            const std::string id = "osm_seg_" + std::to_string(i) + "_" + std::to_string(j);
            viewer->addLine(pa, pb,
                            static_cast<double>(c.r) / 255.0,
                            static_cast<double>(c.g) / 255.0,
                            static_cast<double>(c.b) / 255.0,
                            id);
            ++segment_count;
        }
    }

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    std::cout << "Map points=" << map_cloud->points.size()
              << ", scans loaded=" << scans_loaded
              << ", scans skipped=" << scans_skipped
              << ", initial_position_xyz=[" << init_rel_pos(0)
              << "," << init_rel_pos(1)
              << "," << init_rel_pos(2) << "]"
              << ", OSM polylines=" << osm_data.polylines.size()
              << ", OSM segments=" << segment_count
              << ", skip_frames=" << skip_frames << std::endl;

    while (!viewer->wasStopped()) {
        viewer->spin();
    }

    return 0;
}

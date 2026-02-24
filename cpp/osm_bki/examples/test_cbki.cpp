/**
 * Continuous mapping example (C++).
 *
 * Mirrors OSM-BKI/python/scripts/continuous_mapping_example.py:
 * - Loads scans and labels from directories
 * - Initializes BKI with OSM prior (binary or XML)
 * - Updates map with each scan; optionally infers and saves refined labels
 * - Optionally saves/loads map state
 * - Visualizes the accumulated point cloud with refined labels in PCL
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "continuous_bki.hpp"
#include "dataset_utils.hpp"
#include "file_io.hpp"

namespace fs = std::filesystem;

namespace {

struct Color {
    uint8_t r, g, b;
};

Color colorFromLabel(uint32_t label) {
    const uint32_t h = label * 2654435761u;
    return Color{
        static_cast<uint8_t>((h >> 16) & 0xFF),
        static_cast<uint8_t>((h >> 8) & 0xFF),
        static_cast<uint8_t>(h & 0xFF),
    };
}

void parseArgs(int argc, char** argv,
               std::string& config_path, std::string& osm_path,
               std::string& scan_dir, std::string& label_dir,
               std::string& output_dir, std::string& map_state,
               bool& no_viz,
               float& osm_prior_strength) {
    config_path = "configs/mcd_config.yaml";
    osm_path.clear();
    scan_dir.clear();
    label_dir.clear();
    output_dir.clear();
    map_state.clear();
    no_viz = false;
    osm_prior_strength = 0.0f;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 < argc) return argv[++i];
            return nullptr;
        };
        if (std::strcmp(arg, "--config") == 0 && next()) config_path = argv[i];
        else if (std::strcmp(arg, "--osm") == 0 && next()) osm_path = argv[i];
        else if (std::strcmp(arg, "--scan-dir") == 0 && next()) scan_dir = argv[i];
        else if (std::strcmp(arg, "--label-dir") == 0 && next()) label_dir = argv[i];
        else if (std::strcmp(arg, "--output-dir") == 0 && next()) output_dir = argv[i];
        else if (std::strcmp(arg, "--map-state") == 0 && next()) map_state = argv[i];
        else if (std::strcmp(arg, "--no-viz") == 0) no_viz = true;
        else if (std::strcmp(arg, "--osm-prior-strength") == 0 && next()) osm_prior_strength = std::stof(argv[i]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path, osm_path, scan_dir, label_dir, output_dir, map_state;
    bool no_viz;
    float osm_prior_strength;
    parseArgs(argc, argv, config_path, osm_path, scan_dir, label_dir,
              output_dir, map_state, no_viz, osm_prior_strength);

    // If paths not given, try loading from config (direct paths or dataset_root_path+sequence).
    if (osm_path.empty() || scan_dir.empty() || label_dir.empty()) {
        continuous_bki::DatasetConfig dataset_config;
        std::string err;
        if (!continuous_bki::loadDatasetConfig(config_path, dataset_config, err)) {
            std::cerr << "Failed to load dataset config from " << config_path << ": " << err << std::endl;
            std::cerr << "Paths can be set in config (lidar_dir, label_dir, osm_file) or via --osm, --scan-dir, --label-dir.\n";
            return 1;
        }
        if (scan_dir.empty()) scan_dir = dataset_config.lidar_dir;
        if (label_dir.empty()) label_dir = dataset_config.label_dir;
        if (osm_path.empty() && !dataset_config.osm_file.empty()) {
            if (!dataset_config.dataset_root_path.empty()) {
                osm_path = (fs::path(dataset_config.dataset_root_path) / dataset_config.osm_file).string();
            } else {
                const auto config_dir = fs::path(config_path).parent_path();
                osm_path = (config_dir / dataset_config.osm_file).lexically_normal().string();
            }
        }
    }

    if (osm_path.empty() || scan_dir.empty() || label_dir.empty()) {
        std::cerr << "Usage: " << (argc ? argv[0] : "test_cbki")
                  << " [--config <yaml>] [--osm <path>] [--scan-dir <dir>] [--label-dir <dir>] [--output-dir <dir>] [--map-state <file>] [--no-viz] [--osm-prior-strength <f>]\n"
                  << "Paths from config (lidar_dir, label_dir, osm_file) or via CLI.\n";
        return 1;
    }

    if (!fs::exists(osm_path)) {
        std::cerr << "OSM file not found: " << osm_path << std::endl;
        return 1;
    }
    if (!fs::exists(scan_dir) || !fs::is_directory(scan_dir)) {
        std::cerr << "Scan directory not found or not a directory: " << scan_dir << std::endl;
        return 1;
    }
    if (!fs::exists(label_dir) || !fs::is_directory(label_dir)) {
        std::cerr << "Label directory not found or not a directory: " << label_dir << std::endl;
        return 1;
    }

    std::cout << "Loading config from " << config_path << "..." << std::endl;
    continuous_bki::Config config;
    try {
        config = continuous_bki::loadConfigFromYAML(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loading OSM from " << osm_path << "..." << std::endl;
    continuous_bki::OSMData osm_data = continuous_bki::loadOSM(osm_path, config);

    const float resolution = 0.1f;
    const float l_scale = 0.5f;
    std::cout << "Initializing BKI (resolution=" << resolution << ", l_scale=" << l_scale << ")..." << std::endl;
    continuous_bki::ContinuousBKI bki(
        config, osm_data,
        resolution, l_scale,
        1.0f, 5.0f, 0.3f,  // sigma_0, prior_delta, height_sigma
        true, true,         // use_semantic_kernel, use_spatial_kernel
        -1,                 // num_threads
        1.0f,              // alpha0
        true,              // seed_osm_prior
        osm_prior_strength,
        true);             // osm_fallback_in_infer

    std::cout << "BKI initialized with " << bki.size() << " voxels" << std::endl;
    
    if (!map_state.empty() && fs::exists(map_state)) {
        std::cout << "Loading map state from " << map_state << "..." << std::endl;
        bki.load(map_state);
        std::cout << "  Loaded " << bki.size() << " voxels" << std::endl;
    }

    std::vector<fs::path> scan_files;
    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin")
            scan_files.push_back(entry.path());
    }
    std::sort(scan_files.begin(), scan_files.end());
    std::cout << "Found " << scan_files.size() << " scans" << std::endl;

    if (scan_files.empty()) {
        std::cerr << "No .bin scans in " << scan_dir << std::endl;
        return 1;
    }

    std::vector<continuous_bki::Point3D> all_points;
    all_points.reserve(scan_files.size() * 50000u);

    for (size_t i = 0; i < scan_files.size(); ++i) {
        const std::string scan_path = scan_files[i].string();
        const std::string stem = scan_files[i].stem().string();

        std::string label_path = (fs::path(label_dir) / (stem + ".label")).string();
        if (!fs::exists(label_path))
            label_path = (fs::path(label_dir) / (stem + ".bin")).string();
        if (!fs::exists(label_path)) {
            std::cerr << "Warning: no label for " << stem << ", skipping" << std::endl;
            continue;
        }

        std::vector<continuous_bki::Point3D> points;
        std::vector<uint32_t> labels;
        if (!continuous_bki::readPointCloudBin(scan_path, points) ||
            !continuous_bki::readLabelBin(label_path, labels) ||
            points.size() != labels.size()) {
            std::cerr << "Warning: failed or size mismatch for " << stem << ", skipping" << std::endl;
            continue;
        }

        std::cout << "[" << (i + 1) << "/" << scan_files.size() << "] " << stem
                  << " (" << points.size() << " points) – updating map..." << std::endl;
        bki.update(labels, points);
        std::cout << "  Map size: " << bki.size() << " voxels" << std::endl;

        if (!output_dir.empty()) {
            std::vector<uint32_t> refined = bki.infer(points);
            const std::string out_path = (fs::path(output_dir) / (stem + "_refined.label")).string();
            std::ofstream out(out_path, std::ios::binary);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(refined.data()), refined.size() * sizeof(uint32_t));
                std::cout << "  Saved " << out_path << std::endl;
            }
        }

        for (const auto& p : points)
            all_points.push_back(p);
    }

    if (!map_state.empty()) {
        std::cout << "Saving map state to " << map_state << "..." << std::endl;
        bki.save(map_state);
        std::cout << "Final map size: " << bki.size() << " voxels" << std::endl;
    }

    std::cout << "Continuous mapping complete. Total points: " << all_points.size() << std::endl;

    if (no_viz || all_points.empty()) {
        std::cout << "Skipping visualization (--no-viz or no points)." << std::endl;
        return 0;
    }

    std::cout << "Running inference on all points for visualization..." << std::endl;
    std::vector<uint32_t> refined_labels = bki.infer(all_points);
    if (refined_labels.size() != all_points.size()) {
        std::cerr << "Infer size mismatch" << std::endl;
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->resize(all_points.size());
    for (size_t i = 0; i < all_points.size(); ++i) {
        cloud->points[i].x = all_points[i].x;
        cloud->points[i].y = all_points[i].y;
        cloud->points[i].z = all_points[i].z;
        const Color c = colorFromLabel(refined_labels[i]);
        cloud->points[i].r = c.r;
        cloud->points[i].g = c.g;
        cloud->points[i].b = c.b;
    }
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Continuous BKI – refined labels"));
    viewer->setBackgroundColor(0.05f, 0.05f, 0.05f);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    std::cout << "Displaying " << cloud->points.size() << " points (refined labels). Close window to exit." << std::endl;
    while (!viewer->wasStopped())
        viewer->spin();

    return 0;
}

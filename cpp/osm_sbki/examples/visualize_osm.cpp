#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "osm_parser.hpp"

namespace {

struct Rgb {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

Rgb colorForWay(const osm_parser::Polyline2D& way) {
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

    auto it = way.tags.find("building");
    if (it != way.tags.end()) return {30, 180, 30};

    it = way.tags.find("highway");
    if (it != way.tags.end()) {
        if (it->second == "footway" || it->second == "path" || it->second == "pedestrian") {
            return {220, 220, 220};
        }
        return {240, 120, 20};
    }

    it = way.tags.find("landuse");
    if (it != way.tags.end()) return {20, 160, 20};

    it = way.tags.find("natural");
    if (it != way.tags.end()) return {20, 130, 20};

    it = way.tags.find("barrier");
    if (it != way.tags.end()) return {170, 120, 70};

    return {140, 140, 140};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string config_path =
        (argc > 1) ? argv[1] : "configs/osm_config.yaml";

    osm_parser::OSMConfig config;
    std::string error_msg;
    if (!osm_parser::loadOSMConfig(config_path, config, error_msg)) {
        std::cerr << "Failed to load OSM config: " << error_msg << std::endl;
        return 1;
    }

    osm_parser::ParsedOSMData parsed;
    if (!osm_parser::parsePolylines(config, parsed, error_msg)) {
        std::cerr << "Failed to parse OSM with osmium: " << error_msg << std::endl;
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    point_cloud->is_dense = false;
    point_cloud->points.reserve(parsed.tagged_points.size());

    for (const auto& pt : parsed.tagged_points) {
        pcl::PointXYZRGB p;
        p.x = pt.first;
        p.y = pt.second;
        p.z = 0.1f;
        p.r = 80;
        p.g = 180;
        p.b = 255;
        point_cloud->points.push_back(p);
    }

    point_cloud->width = static_cast<uint32_t>(point_cloud->points.size());
    point_cloud->height = 1;

    std::cout << "Loaded OSM polylines=" << parsed.polylines.size()
              << ", tagged points=" << parsed.tagged_points.size()
              << ", origin(lat,lon)=[" << parsed.origin_lat << ", " << parsed.origin_lon << "]"
              << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("OSM Viewer"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    int segment_count = 0;
    for (size_t i = 0; i < parsed.polylines.size(); ++i) {
        const auto& way = parsed.polylines[i];
        if (way.points.size() < 2) continue;

        const Rgb color = colorForWay(way);
        for (size_t j = 0; j + 1 < way.points.size(); ++j) {
            const auto& a = way.points[j];
            const auto& b = way.points[j + 1];
            pcl::PointXYZ pa(a.first, a.second, 0.0f);
            pcl::PointXYZ pb(b.first, b.second, 0.0f);
            const std::string id = "seg_" + std::to_string(i) + "_" + std::to_string(j);
            viewer->addLine(pa, pb,
                            static_cast<double>(color.r) / 255.0,
                            static_cast<double>(color.g) / 255.0,
                            static_cast<double>(color.b) / 255.0,
                            id);
            ++segment_count;
        }

        // If closed and not explicitly ended at first point, close it visually.
        if (way.is_closed && way.points.size() > 2) {
            const auto& first = way.points.front();
            const auto& last = way.points.back();
            if (std::fabs(first.first - last.first) > 1e-4f || std::fabs(first.second - last.second) > 1e-4f) {
                pcl::PointXYZ pa(last.first, last.second, 0.0f);
                pcl::PointXYZ pb(first.first, first.second, 0.0f);
                const std::string id = "seg_close_" + std::to_string(i);
                viewer->addLine(pa, pb,
                                static_cast<double>(color.r) / 255.0,
                                static_cast<double>(color.g) / 255.0,
                                static_cast<double>(color.b) / 255.0,
                                id);
                ++segment_count;
            }
        }
    }

    if (!point_cloud->points.empty()) {
        viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud, "osm_points");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "osm_points");
    }

    std::cout << "Rendered OSM line segments=" << segment_count << std::endl;
    viewer->addCoordinateSystem(10.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spin();
    }

    return 0;
}

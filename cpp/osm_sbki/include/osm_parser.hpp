#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace osm_parser {

enum class Category {
    Building,
    Road,
    Sidewalk,
    Parking,
    Fence,
    Stairs,
    Grassland,
    Tree,
    Unknown,
};

struct OSMConfig {
    std::string osm_file;
    bool include_buildings = true;
    bool include_roads = true;
    bool include_sidewalks = true;
    bool include_parking = true;
    bool include_fences = true;
    bool include_stairs = true;
    bool include_grasslands = true;
    bool include_trees = true;
};

struct Polyline2D {
    std::vector<std::pair<float, float>> points;
    std::map<std::string, std::string> tags;
    bool is_closed = false;
    Category category = Category::Unknown;
};

struct ParsedOSMData {
    double origin_lat = 0.0;
    double origin_lon = 0.0;
    std::vector<Polyline2D> polylines;
    std::vector<std::pair<float, float>> tagged_points;
};

bool loadOSMConfig(const std::string& config_path, OSMConfig& config, std::string& error_msg);

// Parse OSM with libosmium into projected 2D filtered geometry (meters).
// origin is computed from node centroid to keep coordinates near the dataset area.
bool parsePolylines(const OSMConfig& config, ParsedOSMData& data, std::string& error_msg);

}  // namespace osm_parser

#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace osm_xml_parser {

// Simple XML parser specifically for OSM format
// Handles <node>, <way>, <tag> elements

constexpr double EARTH_RADIUS = 6378137.0;

inline double deg_to_rad(double deg) {
    return deg * M_PI / 180.0;
}

// Scaled Mercator projection (matches KITTI / MCD pipeline)
// This produces coordinates consistent with the lidar2osm pipeline's
// latlon_to_mercator() function.
struct MercatorProjection {
    double scale;     // cos(origin_lat_rad)
    double ox, oy;    // Mercator coords of origin
    double offset_x;  // World-frame X of the GPS origin point
    double offset_y;  // World-frame Y of the GPS origin point

    MercatorProjection()
        : scale(1.0), ox(0.0), oy(0.0), offset_x(0.0), offset_y(0.0) {}

    void setOrigin(double origin_lat, double origin_lon,
                   double world_offset_x = 0.0, double world_offset_y = 0.0) {
        scale = std::cos(deg_to_rad(origin_lat));
        ox = scale * origin_lon * M_PI * EARTH_RADIUS / 180.0;
        oy = scale * EARTH_RADIUS * std::log(std::tan((90.0 + origin_lat) * M_PI / 360.0));
        offset_x = world_offset_x;
        offset_y = world_offset_y;
    }

    // Convert lat/lon to world-frame (x, y)
    std::pair<double, double> toWorld(double lat, double lon) const {
        double mx = scale * lon * M_PI * EARTH_RADIUS / 180.0;
        double my = scale * EARTH_RADIUS * std::log(std::tan((90.0 + lat) * M_PI / 360.0));
        double x = (mx - ox) + offset_x;
        double y = (my - oy) + offset_y;
        return {x, y};
    }
};

// Simple flat-earth projection (fallback when no GPS origin is configured)
inline std::pair<double, double> latlon_to_meters(double lat, double lon, double origin_lat, double origin_lon) {
    double x = deg_to_rad(lon - origin_lon) * std::cos(deg_to_rad(origin_lat)) * EARTH_RADIUS;
    double y = deg_to_rad(lat - origin_lat) * EARTH_RADIUS;
    return {x, y};
}

inline std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

// Extract XML attribute value, checking for word boundary to avoid
// substring matches (e.g. "id=" must not match inside "uid=").
inline std::string get_attribute(const std::string& line, const std::string& attr) {
    std::string search = attr + "=\"";
    size_t pos = 0;
    while (true) {
        pos = line.find(search, pos);
        if (pos == std::string::npos) return "";
        // Check that the character before is a word boundary (space, <, or start of string)
        if (pos == 0 || line[pos - 1] == ' ' || line[pos - 1] == '<' || line[pos - 1] == '\t') {
            // Valid match
            size_t val_start = pos + search.length();
            size_t val_end = line.find("\"", val_start);
            if (val_end == std::string::npos) return "";
            return line.substr(val_start, val_end - val_start);
        }
        // False match (e.g. "uid" matching "id"), skip and continue
        pos += search.length();
    }
}

struct OSMNode {
    std::string id;
    double lat, lon;
    std::map<std::string, std::string> tags;
};

struct OSMWay {
    std::string id;
    std::vector<std::string> node_refs;
    std::map<std::string, std::string> tags;
    bool is_closed() const {
        return node_refs.size() >= 3 && node_refs.front() == node_refs.back();
    }
};

class OSMParser {
public:
    std::map<std::string, OSMNode> nodes;
    std::vector<OSMWay> ways;
    
    void parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open OSM file: " + filename);
        }
        
        std::string line;
        OSMNode* current_node = nullptr;
        OSMWay* current_way = nullptr;
        
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            
            if (trimmed.find("<node ") != std::string::npos) {
                std::string id = get_attribute(trimmed, "id");
                std::string lat_str = get_attribute(trimmed, "lat");
                std::string lon_str = get_attribute(trimmed, "lon");
                
                if (!id.empty() && !lat_str.empty() && !lon_str.empty()) {
                    OSMNode node;
                    node.id = id;
                    node.lat = std::stod(lat_str);
                    node.lon = std::stod(lon_str);
                    
                    if (trimmed.find("/>") != std::string::npos) {
                        // Self-closing tag
                        nodes[id] = node;
                    } else {
                        // Has children (tags)
                        nodes[id] = node;
                        current_node = &nodes[id];
                    }
                }
            }
            else if (trimmed.find("</node>") != std::string::npos) {
                current_node = nullptr;
            }
            else if (trimmed.find("<way ") != std::string::npos) {
                OSMWay way;
                way.id = get_attribute(trimmed, "id");
                ways.push_back(way);
                current_way = &ways.back();
            }
            else if (trimmed.find("</way>") != std::string::npos) {
                current_way = nullptr;
            }
            else if (trimmed.find("<nd ") != std::string::npos && current_way) {
                std::string ref = get_attribute(trimmed, "ref");
                if (!ref.empty()) {
                    current_way->node_refs.push_back(ref);
                }
            }
            else if (trimmed.find("<tag ") != std::string::npos) {
                std::string k = get_attribute(trimmed, "k");
                std::string v = get_attribute(trimmed, "v");
                if (!k.empty()) {
                    if (current_node) {
                        current_node->tags[k] = v;
                    } else if (current_way) {
                        current_way->tags[k] = v;
                    }
                }
            }
        }
        
        file.close();
    }
    
    std::pair<double, double> get_center() const {
        if (nodes.empty()) return {0.0, 0.0};
        
        double lat_sum = 0.0, lon_sum = 0.0;
        int count = 0;
        for (const auto& kv : nodes) {
            lat_sum += kv.second.lat;
            lon_sum += kv.second.lon;
            count++;
        }
        return {lat_sum / count, lon_sum / count};
    }
};

} // namespace osm_xml_parser

#include "osm_loader.hpp"
#include "continuous_bki.hpp"
#include "osm_xml_parser.hpp"
#include "yaml_parser.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace continuous_bki {

// --- Helper Functions ---

// Classify a way's tags into an OSM class index.
static int classifyWayTags(const std::map<std::string, std::string>& tags,
                           const std::map<std::string, int>& osm_class_map,
                           bool& is_area) {
    is_area = false;

    auto it = tags.find("building");
    if (it != tags.end()) {
        is_area = true;
        auto c = osm_class_map.find("buildings");
        return (c != osm_class_map.end()) ? c->second : -1;
    }

    it = tags.find("highway");
    if (it != tags.end()) {
        const std::string& val = it->second;
        is_area = false;
        if (val == "footway" || val == "path" || val == "steps" || val == "pedestrian") {
            auto c = osm_class_map.find("sidewalks");
            return (c != osm_class_map.end()) ? c->second : -1;
        } else {
            auto c = osm_class_map.find("roads");
            return (c != osm_class_map.end()) ? c->second : -1;
        }
    }

    it = tags.find("landuse");
    if (it != tags.end()) {
        is_area = true;
        const std::string& val = it->second;
        if (val == "grass" || val == "meadow" || val == "park" || val == "recreation_ground") {
            auto c = osm_class_map.find("grasslands");
            return (c != osm_class_map.end()) ? c->second : -1;
        } else if (val == "forest") {
            auto c = osm_class_map.find("trees");
            return (c != osm_class_map.end()) ? c->second : -1;
        }
        return -1;
    }

    it = tags.find("natural");
    if (it != tags.end()) {
        is_area = true;
        const std::string& val = it->second;
        if (val == "tree" || val == "wood") {
            auto c = osm_class_map.find("trees");
            return (c != osm_class_map.end()) ? c->second : -1;
        } else if (val == "grassland" || val == "scrub") {
            auto c = osm_class_map.find("grasslands");
            return (c != osm_class_map.end()) ? c->second : -1;
        }
        return -1;
    }

    it = tags.find("barrier");
    if (it != tags.end()) {
        is_area = false;
        const std::string& val = it->second;
        if (val == "fence" || val == "wall" || val == "hedge") {
            auto c = osm_class_map.find("fences");
            return (c != osm_class_map.end()) ? c->second : -1;
        }
        return -1;
    }

    it = tags.find("amenity");
    if (it != tags.end()) {
        is_area = true;
        if (it->second == "parking") {
            auto c = osm_class_map.find("parking");
            return (c != osm_class_map.end()) ? c->second : -1;
        }
        return -1;
    }

    return -1;
}

static Polygon bufferPolyline(const std::vector<Point2D>& coords, float half_width) {
    Polygon poly;
    if (coords.size() < 2) return poly;

    std::vector<Point2D> left_side, right_side;
    for (size_t i = 0; i < coords.size() - 1; i++) {
        float dx = coords[i+1].x - coords[i].x;
        float dy = coords[i+1].y - coords[i].y;
        float len = std::sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) continue;
        float nx = -dy / len * half_width;
        float ny = dx / len * half_width;

        if (i == 0) {
            left_side.push_back(Point2D(coords[i].x + nx, coords[i].y + ny));
            right_side.push_back(Point2D(coords[i].x - nx, coords[i].y - ny));
        }
        left_side.push_back(Point2D(coords[i+1].x + nx, coords[i+1].y + ny));
        right_side.push_back(Point2D(coords[i+1].x - nx, coords[i+1].y - ny));
    }

    for (const auto& p : left_side) poly.points.push_back(p);
    for (auto it = right_side.rbegin(); it != right_side.rend(); ++it) {
        poly.points.push_back(*it);
    }

    if (poly.points.size() >= 3) {
        poly.computeBounds();
    }
    return poly;
}

// --- Loader Implementations ---

OSMData loadOSMBinary(const std::string& filename,
                      const std::map<std::string, int>& osm_class_map,
                      const std::vector<std::string>& osm_categories) {
    OSMData data;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OSM file: " + filename);
    }

    if (osm_categories.empty()) {
        throw std::runtime_error("OSM categories missing from config.");
    }

    for (const auto& cat : osm_categories) {
        uint32_t num_items;
        file.read(reinterpret_cast<char*>(&num_items), sizeof(uint32_t));
        if (!file.good()) break;

        auto class_it = osm_class_map.find(cat);
        bool has_class = (class_it != osm_class_map.end());
        int class_idx = has_class ? class_it->second : -1;

        for (uint32_t i = 0; i < num_items; i++) {
            uint32_t n_pts;
            file.read(reinterpret_cast<char*>(&n_pts), sizeof(uint32_t));
            if (!file.good()) break;

            Polygon poly;
            poly.points.reserve(n_pts);
            for (uint32_t j = 0; j < n_pts; j++) {
                float x, y;
                file.read(reinterpret_cast<char*>(&x), sizeof(float));
                file.read(reinterpret_cast<char*>(&y), sizeof(float));
                poly.points.push_back(Point2D(x, y));
            }
            poly.computeBounds();
            if (has_class) {
                data.geometries[class_idx].push_back(poly);
            }
        }
    }

    file.close();
    return data;
}

OSMData loadOSMXML(const std::string& filename,
                   const Config& config) {
    OSMData data;
    osm_xml_parser::OSMParser parser;
    
    const auto& osm_class_map = config.osm_class_map;

    try {
        parser.parse(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse OSM XML: " + std::string(e.what()));
    }
    
    if (parser.nodes.empty()) {
        std::cerr << "Warning: No nodes found in OSM file" << std::endl;
        return data;
    }
    
    double origin_lat, origin_lon;
    double offset_x = config.osm_world_offset_x;
    double offset_y = config.osm_world_offset_y;

    if (config.has_osm_origin) {
        origin_lat = config.osm_origin_lat;
        origin_lon = config.osm_origin_lon;
        std::cout << "OSM XML: Using config origin ("
                  << origin_lat << ", " << origin_lon << ")" << std::endl;
    } else {
        std::ifstream bounds_scan(filename);
        std::string bline;
        bool found_bounds = false;
        while (std::getline(bounds_scan, bline)) {
            if (bline.find("<bounds") != std::string::npos) {
                std::string minlat_s = osm_xml_parser::get_attribute(bline, "minlat");
                std::string maxlat_s = osm_xml_parser::get_attribute(bline, "maxlat");
                std::string minlon_s = osm_xml_parser::get_attribute(bline, "minlon");
                std::string maxlon_s = osm_xml_parser::get_attribute(bline, "maxlon");
                if (!minlat_s.empty() && !maxlat_s.empty() &&
                    !minlon_s.empty() && !maxlon_s.empty()) {
                    double minlat = std::stod(minlat_s);
                    double maxlat = std::stod(maxlat_s);
                    double minlon = std::stod(minlon_s);
                    double maxlon = std::stod(maxlon_s);
                    origin_lat = (minlat + maxlat) / 2.0;
                    origin_lon = (minlon + maxlon) / 2.0;
                    found_bounds = true;
                    std::cout << "OSM XML: Using <bounds> centroid as origin ("
                              << origin_lat << ", " << origin_lon << ")" << std::endl;
                }
                break;
            }
        }
        if (!found_bounds) {
            auto center = parser.get_center();
            origin_lat = center.first;
            origin_lon = center.second;
            std::cout << "OSM XML: No <bounds> found, using node centroid as origin ("
                      << origin_lat << ", " << origin_lon << ")" << std::endl;
        }
    }

    std::cout << "OSM XML: Mercator projection, origin ("
              << origin_lat << ", " << origin_lon
              << "), world offset (" << offset_x << ", " << offset_y << ")" << std::endl;

    std::map<std::string, Point2D> node_coords;

    for (const auto& kv : parser.nodes) {
        auto xy = osm_xml_parser::latlon_to_mercator(
            kv.second.lat, kv.second.lon, origin_lat, origin_lon, offset_x, offset_y);
        node_coords[kv.first] = Point2D(static_cast<float>(xy.first), static_cast<float>(xy.second));
    }
    
    for (const auto& kv : parser.nodes) {
        const osm_xml_parser::OSMNode& node = kv.second;
        const Point2D& pt = node_coords[kv.first];
        
        int class_idx = -1;
        for (const auto& tag : node.tags) {
            const std::string& key = tag.first;
            const std::string& val = tag.second;
            
            if (key == "highway") {
                if (val == "street_lamp" || val == "street_light") {
                    auto it = osm_class_map.find("poles");
                    if (it != osm_class_map.end()) class_idx = it->second;
                } else if (val == "traffic_signals" || val == "stop") {
                    auto it = osm_class_map.find("traffic_signs");
                    if (it != osm_class_map.end()) class_idx = it->second;
                }
            } else if (key == "barrier") {
                if (val == "bollard" || val == "gate") {
                    auto it = osm_class_map.find("barriers");
                    if (it != osm_class_map.end()) class_idx = it->second;
                }
            } else if (key == "amenity" && val == "parking") {
                auto it = osm_class_map.find("parking");
                if (it != osm_class_map.end()) class_idx = it->second;
            }
            
            if (class_idx >= 0) {
                data.point_features[class_idx].push_back(pt);
                break;
            }
        }
    }
    
    constexpr float ROAD_HALF_WIDTH = 3.0f;
    constexpr float SIDEWALK_HALF_WIDTH = 1.5f;
    constexpr float FENCE_HALF_WIDTH = 0.3f;

    int polygon_count = 0, polyline_count = 0;
    for (const auto& way : parser.ways) {
        if (way.node_refs.size() < 2) continue;
        
        std::vector<Point2D> coords;
        for (const auto& ref : way.node_refs) {
            auto it = node_coords.find(ref);
            if (it != node_coords.end()) {
                coords.push_back(it->second);
            }
        }
        
        if (coords.size() < 2) continue;
        
        bool is_area = false;
        int class_idx = classifyWayTags(way.tags, osm_class_map, is_area);
        if (class_idx < 0) continue;
        
        bool way_is_closed = way.is_closed();

        if (is_area && way_is_closed && coords.size() >= 3) {
            Polygon poly;
            poly.points = coords;
            poly.computeBounds();
            data.geometries[class_idx].push_back(poly);
            polygon_count++;
        } else if (!is_area || !way_is_closed) {
            float hw = ROAD_HALF_WIDTH;

            auto hw_it = way.tags.find("highway");
            if (hw_it != way.tags.end()) {
                const std::string& v = hw_it->second;
                if (v == "footway" || v == "path" || v == "steps" || v == "pedestrian") {
                    hw = SIDEWALK_HALF_WIDTH;
                }
            }
            auto bar_it = way.tags.find("barrier");
            if (bar_it != way.tags.end()) {
                hw = FENCE_HALF_WIDTH;
            }

            Polygon buffered = bufferPolyline(coords, hw);
            if (buffered.points.size() >= 3) {
                data.geometries[class_idx].push_back(buffered);
                polyline_count++;
            }
        }
    }
    
    int total_point_features = 0;
    for (const auto& kv : data.point_features) {
        total_point_features += static_cast<int>(kv.second.size());
    }
    std::cout << "Loaded OSM XML: " 
              << polygon_count << " polygons, "
              << polyline_count << " buffered polylines, "
              << total_point_features << " point features" << std::endl;
    
    return data;
}

OSMData loadOSM(const std::string& filename,
                const Config& config) {
    if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".osm") {
        return loadOSMXML(filename, config);
    } else {
        return loadOSMBinary(filename, config.osm_class_map, config.osm_categories);
    }
}

Config loadConfigFromYAML(const std::string& config_path) {
    Config config;
    try {
        yaml_parser::YAMLNode yaml;
        yaml.parseFile(config_path);

        config.labels = yaml.getLabels();
        config.confusion_matrix = yaml.getConfusionMatrix();
        config.osm_class_map = yaml.getOSMClassMap();
        config.osm_categories = yaml.getOSMCategories();

        auto height_filter_str = yaml.getOSMHeightFilter();
        for (const auto& kv : height_filter_str) {
            auto it = config.osm_class_map.find(kv.first);
            if (it != config.osm_class_map.end()) {
                config.height_filter_map[it->second] = kv.second;
            }
        }

        std::vector<int> all_classes;
        for (const auto& kv : config.labels) {
            all_classes.push_back(kv.first);
        }
        for (size_t i = 0; i < all_classes.size(); i++) {
            config.raw_to_dense[all_classes[i]] = static_cast<int>(i);
            config.dense_to_raw[static_cast<int>(i)] = all_classes[i];
        }
        config.num_total_classes = static_cast<int>(all_classes.size());

        auto scalar_it = yaml.scalars.find("osm_origin_lat");
        if (scalar_it != yaml.scalars.end()) {
            config.osm_origin_lat = std::stod(scalar_it->second);
            auto lon_it = yaml.scalars.find("osm_origin_lon");
            if (lon_it != yaml.scalars.end()) {
                config.osm_origin_lon = std::stod(lon_it->second);
                config.has_osm_origin = true;
            }
        }
        if (!config.has_osm_origin) {
            auto init_it = yaml.scalars.find("init_latlon_day_06");
            if (init_it != yaml.scalars.end()) {
                std::string s = init_it->second;
                size_t a = s.find('['), b = s.find(','), c = s.find(']');
                if (a != std::string::npos && b != std::string::npos && c != std::string::npos) {
                    try {
                        config.osm_origin_lat = std::stod(s.substr(a + 1, b - a - 1));
                        config.osm_origin_lon = std::stod(s.substr(b + 1, c - b - 1));
                        config.has_osm_origin = true;
                    } catch (...) { /* leave has_osm_origin false */ }
                }
            }
        }
        auto offset_x_it = yaml.scalars.find("osm_world_offset_x");
        if (offset_x_it != yaml.scalars.end()) {
            config.osm_world_offset_x = std::stod(offset_x_it->second);
        }
        auto offset_y_it = yaml.scalars.find("osm_world_offset_y");
        if (offset_y_it != yaml.scalars.end()) {
            config.osm_world_offset_y = std::stod(offset_y_it->second);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error loading config from " << config_path << ": " << e.what() << std::endl;
        throw;
    }
    return config;
}

} // namespace continuous_bki

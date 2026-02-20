#include "osm_parser.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include <osmium/handler.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/io/any_input.hpp>
#include <osmium/osm/area.hpp>
#include <osmium/osm/location.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/tag.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/tags/tags_filter.hpp>
#include <osmium/visitor.hpp>

namespace osm_parser {

namespace {

std::string trim(const std::string& s) {
    const size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool parseBool(const std::string& v, bool& out) {
    const std::string t = trim(v);
    if (t == "true" || t == "True" || t == "1") {
        out = true;
        return true;
    }
    if (t == "false" || t == "False" || t == "0") {
        out = false;
        return true;
    }
    return false;
}

std::string stripQuotes(const std::string& s) {
    const std::string t = trim(s);
    if (t.size() >= 2) {
        const char a = t.front();
        const char b = t.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            return t.substr(1, t.size() - 2);
        }
    }
    return t;
}

class OriginEstimator : public osmium::handler::Handler {
public:
    void node(const osmium::Node& node) {
        if (!node.location().valid()) return;
        lat_sum_ += node.location().lat();
        lon_sum_ += node.location().lon();
        ++count_;
    }

    bool hasData() const { return count_ > 0; }
    double originLat() const { return lat_sum_ / static_cast<double>(count_); }
    double originLon() const { return lon_sum_ / static_cast<double>(count_); }

private:
    double lat_sum_ = 0.0;
    double lon_sum_ = 0.0;
    size_t count_ = 0;
};

class PolylineHandler : public osmium::handler::Handler {
public:
    PolylineHandler(double origin_lat, double origin_lon, const OSMConfig& config, ParsedOSMData& out)
        : config_(config), out_(out) {
        meters_per_deg_lat_ = 110540.0;
        meters_per_deg_lon_ = 111320.0 * std::cos(origin_lat * M_PI / 180.0);
        origin_lat_ = origin_lat;
        origin_lon_ = origin_lon;
    }

    void node(const osmium::Node& node) {
        if (!config_.include_trees) return;
        if (!node.location().valid()) return;
        const char* natural_tag = node.tags()["natural"];
        if (!natural_tag || std::string(natural_tag) != "tree") return;
        const float x = static_cast<float>((node.location().lon() - origin_lon_) * meters_per_deg_lon_);
        const float y = static_cast<float>((node.location().lat() - origin_lat_) * meters_per_deg_lat_);
        out_.tagged_points.push_back({x, y});
    }

    void way(const osmium::Way& way) {
        Polyline2D geom;
        geom.points.reserve(way.nodes().size());
        for (const auto& node_ref : way.nodes()) {
            const osmium::Location& loc = node_ref.location();
            if (!loc.valid()) continue;
            const float x = static_cast<float>((loc.lon() - origin_lon_) * meters_per_deg_lon_);
            const float y = static_cast<float>((loc.lat() - origin_lat_) * meters_per_deg_lat_);
            geom.points.push_back({x, y});
        }
        if (geom.points.size() < 2) return;

        for (const auto& tag : way.tags()) {
            geom.tags[std::string(tag.key())] = std::string(tag.value());
        }
        geom.is_closed = !way.nodes().empty() && way.nodes().front().ref() == way.nodes().back().ref();

        const char* building_tag = way.tags()["building"];
        if (building_tag && config_.include_buildings && geom.points.size() >= 3) {
            geom.category = Category::Building;
            out_.polylines.push_back(std::move(geom));
            return;
        }

        const char* amenity_tag = way.tags()["amenity"];
        if (amenity_tag && std::string(amenity_tag) == "parking" && config_.include_parking) {
            geom.category = Category::Parking;
            out_.polylines.push_back(std::move(geom));
            return;
        }
        if (amenity_tag && std::string(amenity_tag) == "parking_space" && config_.include_parking) {
            geom.category = Category::Parking;
            out_.polylines.push_back(std::move(geom));
            return;
        }

        const char* barrier_tag = way.tags()["barrier"];
        if (barrier_tag && std::string(barrier_tag) == "fence" && config_.include_fences) {
            geom.category = Category::Fence;
            out_.polylines.push_back(std::move(geom));
            return;
        }

        const char* highway_tag = way.tags()["highway"];
        if (highway_tag && std::string(highway_tag) == "steps" && config_.include_stairs) {
            geom.category = Category::Stairs;
            out_.polylines.push_back(std::move(geom));
            return;
        }

        const char* footway_tag = way.tags()["footway"];
        if (highway_tag && std::string(highway_tag) == "footway" &&
            footway_tag && std::string(footway_tag) == "sidewalk" &&
            config_.include_sidewalks) {
            geom.category = Category::Sidewalk;
            out_.polylines.push_back(std::move(geom));
            return;
        }

        if (highway_tag && config_.include_roads) {
            std::string highway(highway_tag);
            if (highway == "motorway" || highway == "trunk" || highway == "primary" ||
                highway == "secondary" || highway == "tertiary" ||
                highway == "unclassified" || highway == "residential" ||
                highway == "motorway_link" || highway == "trunk_link" ||
                highway == "primary_link" || highway == "secondary_link" ||
                highway == "tertiary_link" || highway == "living_street" ||
                highway == "service" || highway == "pedestrian" ||
                highway == "road" || highway == "cycleway" ||
                highway == "footway" || highway == "path" || highway == "foot") {
                geom.category = Category::Road;
                out_.polylines.push_back(geom);

                const char* sidewalk_tag = way.tags()["sidewalk"];
                if (sidewalk_tag && config_.include_sidewalks) {
                    std::string sw(sidewalk_tag);
                    if (sw == "both" || sw == "left" || sw == "right" || sw == "yes") {
                        Polyline2D sw_geom = geom;
                        sw_geom.category = Category::Sidewalk;
                        out_.polylines.push_back(std::move(sw_geom));
                    }
                }
                return;
            }
        }

        const char* landuse_tag = way.tags()["landuse"];
        if (landuse_tag) {
            std::string landuse(landuse_tag);
            if ((landuse == "grass" || landuse == "meadow" || landuse == "greenfield") &&
                config_.include_grasslands && geom.points.size() >= 3) {
                geom.category = Category::Grassland;
                out_.polylines.push_back(std::move(geom));
                return;
            }
            if (landuse == "forest" && config_.include_trees && geom.points.size() >= 3) {
                geom.category = Category::Tree;
                out_.polylines.push_back(std::move(geom));
                return;
            }
        }

        const char* natural_tag = way.tags()["natural"];
        if (natural_tag) {
            const std::string natural(natural_tag);
            if ((natural == "grassland" || natural == "heath" || natural == "scrub") &&
                config_.include_grasslands && geom.points.size() >= 3) {
                geom.category = Category::Grassland;
                out_.polylines.push_back(std::move(geom));
                return;
            }
            if ((natural == "wood" || natural == "forest") &&
                config_.include_trees && geom.points.size() >= 3) {
                geom.category = Category::Tree;
                out_.polylines.push_back(std::move(geom));
                return;
            }
        }
    }

    void area(const osmium::Area& area) {
        auto push_outer_rings = [&](Category category) {
            for (const auto& outer_ring : area.outer_rings()) {
                Polyline2D geom;
                for (const auto& node_ref : outer_ring) {
                    const osmium::Location& loc = node_ref.location();
                    if (!loc.valid()) continue;
                    const float x = static_cast<float>((loc.lon() - origin_lon_) * meters_per_deg_lon_);
                    const float y = static_cast<float>((loc.lat() - origin_lat_) * meters_per_deg_lat_);
                    geom.points.push_back({x, y});
                }
                if (geom.points.size() >= 3) {
                    for (const auto& tag : area.tags()) {
                        geom.tags[std::string(tag.key())] = std::string(tag.value());
                    }
                    geom.is_closed = true;
                    geom.category = category;
                    out_.polylines.push_back(std::move(geom));
                }
            }
        };

        const char* amenity_tag = area.tags()["amenity"];
        if (amenity_tag && std::string(amenity_tag) == "parking" && config_.include_parking) {
            push_outer_rings(Category::Parking);
            return;
        }

        const char* building_tag = area.tags()["building"];
        if (building_tag && config_.include_buildings) {
            push_outer_rings(Category::Building);
            return;
        }

        const char* landuse_tag = area.tags()["landuse"];
        if (landuse_tag) {
            const std::string landuse(landuse_tag);
            if ((landuse == "grass" || landuse == "meadow" || landuse == "greenfield") &&
                config_.include_grasslands) {
                push_outer_rings(Category::Grassland);
                return;
            }
            if (landuse == "forest" && config_.include_trees) {
                push_outer_rings(Category::Tree);
                return;
            }
        }

        const char* natural_tag = area.tags()["natural"];
        if (natural_tag) {
            const std::string natural(natural_tag);
            if ((natural == "grassland" || natural == "heath" || natural == "scrub") &&
                config_.include_grasslands) {
                push_outer_rings(Category::Grassland);
                return;
            }
            if ((natural == "wood" || natural == "forest") && config_.include_trees) {
                push_outer_rings(Category::Tree);
                return;
            }
        }
    }

private:
    const OSMConfig& config_;
    ParsedOSMData& out_;
    double origin_lat_ = 0.0;
    double origin_lon_ = 0.0;
    double meters_per_deg_lat_ = 0.0;
    double meters_per_deg_lon_ = 0.0;
};

}  // namespace

bool loadOSMConfig(const std::string& config_path, OSMConfig& config, std::string& error_msg) {
    std::ifstream in(config_path);
    if (!in.is_open()) {
        error_msg = "Unable to open OSM config: " + config_path;
        return false;
    }

    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        const size_t c = t.find(':');
        if (c == std::string::npos) continue;
        const std::string key = trim(t.substr(0, c));
        const std::string value = trim(t.substr(c + 1));
        kv[key] = value;
    }

    if (kv.find("osm_file") == kv.end()) {
        error_msg = "Missing required key osm_file in config.";
        return false;
    }
    config.osm_file = stripQuotes(kv["osm_file"]);

    auto parse_flag = [&](const std::string& key, bool& dst) -> bool {
        auto it = kv.find(key);
        if (it == kv.end()) return true;
        if (!parseBool(it->second, dst)) {
            error_msg = "Invalid bool for key " + key + ": " + it->second;
            return false;
        }
        return true;
    };

    return parse_flag("include_buildings", config.include_buildings) &&
           parse_flag("include_roads", config.include_roads) &&
           parse_flag("include_sidewalks", config.include_sidewalks) &&
           parse_flag("include_parking", config.include_parking) &&
           parse_flag("include_fences", config.include_fences) &&
           parse_flag("include_stairs", config.include_stairs) &&
           parse_flag("include_grasslands", config.include_grasslands) &&
           parse_flag("include_trees", config.include_trees);
}

bool parsePolylines(const OSMConfig& config, ParsedOSMData& data, std::string& error_msg) {
    data = ParsedOSMData{};

    try {
        {
            osmium::io::Reader reader{osmium::io::File(config.osm_file)};
            OriginEstimator estimator;
            osmium::apply(reader, estimator);
            reader.close();

            if (!estimator.hasData()) {
                error_msg = "No valid nodes found while estimating OSM origin.";
                return false;
            }
            data.origin_lat = estimator.originLat();
            data.origin_lon = estimator.originLon();
        }

        using Index = osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>;
        Index index;
        osmium::handler::NodeLocationsForWays<Index> loc_handler(index);
        loc_handler.ignore_errors();

        PolylineHandler handler(data.origin_lat, data.origin_lon, config, data);

        osmium::area::AssemblerConfig assembler_config;
        osmium::TagsFilter filter(false);
        filter.add_rule(true, "building");
        filter.add_rule(true, "amenity", "parking");
        filter.add_rule(true, "landuse", "grass");
        filter.add_rule(true, "landuse", "meadow");
        filter.add_rule(true, "landuse", "greenfield");
        filter.add_rule(true, "landuse", "forest");
        filter.add_rule(true, "natural", "grassland");
        filter.add_rule(true, "natural", "heath");
        filter.add_rule(true, "natural", "scrub");
        filter.add_rule(true, "natural", "wood");
        filter.add_rule(true, "natural", "forest");

        using MultipolygonManager = osmium::area::MultipolygonManager<osmium::area::Assembler>;
        MultipolygonManager mp_manager{assembler_config, filter};

        osmium::io::File input_file(config.osm_file);
        osmium::io::Reader reader{input_file};
        osmium::apply(reader, loc_handler, handler, mp_manager);
        reader.close();

        // Required by libosmium before second-pass member lookup.
        mp_manager.prepare_for_lookup();

        osmium::io::Reader reader2{input_file};
        osmium::apply(reader2, loc_handler, mp_manager.handler([&handler](osmium::memory::Buffer&& buffer) {
            for (const auto& item : buffer) {
                if (item.type() == osmium::item_type::area) {
                    handler.area(static_cast<const osmium::Area&>(item));
                }
            }
        }));
        reader2.close();

        if (data.polylines.empty()) {
            error_msg = "No filtered polylines parsed for enabled categories.";
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        error_msg = std::string("OSM parse failed: ") + e.what();
        return false;
    }
}

}  // namespace osm_parser

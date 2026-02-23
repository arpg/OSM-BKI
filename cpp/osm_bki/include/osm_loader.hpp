#pragma once

#include "continuous_bki.hpp"
#include <string>
#include <vector>
#include <map>

namespace continuous_bki {

// Forward declarations
struct OSMData;
struct Config;

// Loaders
OSMData loadOSMBinary(const std::string& filename,
                      const std::map<std::string, int>& osm_class_map,
                      const std::vector<std::string>& osm_categories);

OSMData loadOSMXML(const std::string& filename,
                   const Config& config);

OSMData loadOSM(const std::string& filename,
                const Config& config);

Config loadConfigFromYAML(const std::string& config_path);

} // namespace continuous_bki

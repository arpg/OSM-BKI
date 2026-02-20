#pragma once

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace yaml_parser {

// Simple YAML parser for our specific config format
class YAMLNode {
public:
    std::map<std::string, std::string> scalars;
    std::map<std::string, std::map<int, std::string>> int_maps;  // For labels
    std::map<std::string, std::map<std::string, int>> str_maps;   // For osm_class_map
    std::map<std::string, std::map<int, int>> int_int_maps;       // For label_to_matrix_idx
    std::map<std::string, std::map<std::string, float>> str_float_maps; // For osm_height_filter
    std::map<std::string, std::vector<std::vector<float>>> matrices;
    std::map<std::string, std::vector<std::string>> str_lists;    // For osm_categories

    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    static std::vector<std::string> split(const std::string& str, char delim) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, delim)) {
            tokens.push_back(trim(token));
        }
        return tokens;
    }

    void parseFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filename);
        }

        std::string line;
        std::string current_key;
        bool in_matrix = false;
        std::vector<std::vector<float>> current_matrix;

        while (std::getline(file, line)) {
            // Check indentation BEFORE trimming
            size_t indent = line.find_first_not_of(" \t");
            std::string trimmed_line = trim(line);
            
            // Skip empty lines and comments
            if (trimmed_line.empty() || trimmed_line[0] == '#') continue;

            // Check for list item
            if (trimmed_line[0] == '-') {
                if (in_matrix) {
                    // Parse matrix row: - [0.85, 0.10, ...]
                    size_t start = trimmed_line.find('[');
                    size_t end = trimmed_line.find(']');
                    if (start != std::string::npos && end != std::string::npos) {
                        std::string values_str = trimmed_line.substr(start + 1, end - start - 1);
                        auto values_strs = split(values_str, ',');
                        std::vector<float> row;
                        for (const auto& val_str : values_strs) {
                            try {
                                row.push_back(std::stof(val_str));
                            } catch (...) { }
                        }
                        current_matrix.push_back(row);
                    }
                } else if (current_key == "osm_categories") {
                    std::string item = trim(trimmed_line.substr(1));
                    if (!item.empty()) {
                        if (item.size() >= 2) {
                            char first = item.front();
                            char last = item.back();
                            if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
                                item = item.substr(1, item.size() - 2);
                            }
                        }
                        str_lists[current_key].push_back(item);
                    }
                }
                continue;
            }

            // Check for key: value
            size_t colon_pos = trimmed_line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trim(trimmed_line.substr(0, colon_pos));
                std::string value = trim(trimmed_line.substr(colon_pos + 1));
                
                if (indent == 0 || indent == std::string::npos) {
                    // Top-level key
                    if (in_matrix && !current_matrix.empty()) {
                        matrices[current_key] = current_matrix;
                        current_matrix.clear();
                    }
                    in_matrix = false;
                    current_key = key;
                    
                    if (!value.empty()) {
                        scalars[key] = value;
                    } else {
                        // Next lines will be nested
                        in_matrix = (key == "confusion_matrix");
                    }
                } else {
                    // Nested key-value
                    if (!value.empty()) {
                        // Try to parse as int key
                        try {
                            int int_key = std::stoi(key);
                            if (current_key == "labels") {
                                int_maps[current_key][int_key] = value;
                            } else if (current_key == "label_to_matrix_idx") {
                                int_int_maps[current_key][int_key] = std::stoi(value);
                            }
                        } catch (...) {
                            // String key
                            if (current_key == "osm_class_map") {
                                str_maps[current_key][key] = std::stoi(value);
                            } else if (current_key == "osm_height_filter") {
                                str_float_maps[current_key][key] = std::stof(value);
                            }
                        }
                    }
                }
            }
        }

        // Save last matrix if any
        if (in_matrix && !current_matrix.empty()) {
            matrices[current_key] = current_matrix;
        }
    }

    std::map<int, std::string> getLabels() const {
        auto it = int_maps.find("labels");
        return (it != int_maps.end()) ? it->second : std::map<int, std::string>();
    }

    std::vector<std::vector<float>> getConfusionMatrix() const {
        auto it = matrices.find("confusion_matrix");
        return (it != matrices.end()) ? it->second : std::vector<std::vector<float>>();
    }

    std::map<int, int> getLabelToMatrixIdx() const {
        auto it = int_int_maps.find("label_to_matrix_idx");
        return (it != int_int_maps.end()) ? it->second : std::map<int, int>();
    }

    std::map<std::string, int> getOSMClassMap() const {
        auto it = str_maps.find("osm_class_map");
        return (it != str_maps.end()) ? it->second : std::map<std::string, int>();
    }

    std::vector<std::string> getOSMCategories() const {
        auto it = str_lists.find("osm_categories");
        return (it != str_lists.end()) ? it->second : std::vector<std::string>();
    }

    std::map<std::string, float> getOSMHeightFilter() const {
        auto it = str_float_maps.find("osm_height_filter");
        return (it != str_float_maps.end()) ? it->second : std::map<std::string, float>();
    }
};

} // namespace yaml_parser

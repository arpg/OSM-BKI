#include "dataset_utils.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>

#include "yaml_parser.hpp"

namespace continuous_bki {

namespace {

std::string stripQuotes(const std::string& value) {
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            return value.substr(1, value.size() - 2);
        }
    }
    return value;
}

bool hasBinExtension(const std::filesystem::path& path) {
    return path.has_extension() && path.extension() == ".bin";
}

}  // namespace

bool loadDatasetConfig(const std::string& config_path, DatasetConfig& config, std::string& error_msg) {
    try {
        yaml_parser::YAMLNode yaml;
        yaml.parseFile(config_path);

        auto root_it = yaml.scalars.find("dataset_root_path");
        auto seq_it = yaml.scalars.find("sequence");
        if (root_it == yaml.scalars.end() || seq_it == yaml.scalars.end()) {
            error_msg = "Missing required keys: dataset_root_path and/or sequence in config.";
            return false;
        }

        auto dataset_it = yaml.scalars.find("dataset_name");
        config.dataset_name = (dataset_it != yaml.scalars.end())
                                  ? stripQuotes(dataset_it->second)
                                  : "mcd";
        config.dataset_root_path = stripQuotes(root_it->second);
        config.sequence = stripQuotes(seq_it->second);
        auto skip_it = yaml.scalars.find("skip_frames");
        config.skip_frames = (skip_it != yaml.scalars.end()) ? std::max(0, std::stoi(stripQuotes(skip_it->second))) : 0;

        std::string dataset_name_lower = config.dataset_name;
        std::transform(dataset_name_lower.begin(), dataset_name_lower.end(), dataset_name_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (dataset_name_lower == "mcd") {
            const std::filesystem::path base =
                std::filesystem::path(config.dataset_root_path) / config.sequence;
            config.lidar_dir = (base / "lidar_bin" / "data").string();
            config.label_dir = (base / "gt_labels").string();
            config.pose_path = (base / "pose_inW.csv").string();
            config.calibration_path = (std::filesystem::path(config.dataset_root_path) / "hhs_calib.yaml").string();
        } else {
            error_msg = "Unsupported dataset_name: " + config.dataset_name + ". Currently supported: mcd";
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        error_msg = std::string("Failed to load dataset config: ") + e.what();
        return false;
    }
}

bool collectScanLabelPairs(const DatasetConfig& config, std::vector<ScanLabelPair>& pairs, std::string& error_msg) {
    pairs.clear();

    const std::filesystem::path lidar_dir(config.lidar_dir);
    const std::filesystem::path label_dir(config.label_dir);
    if (!std::filesystem::exists(lidar_dir)) {
        error_msg = "Lidar directory does not exist: " + lidar_dir.string();
        return false;
    }
    if (!std::filesystem::exists(label_dir)) {
        error_msg = "Label directory does not exist: " + label_dir.string();
        return false;
    }

    std::vector<std::filesystem::path> lidar_files;
    for (const auto& entry : std::filesystem::directory_iterator(lidar_dir)) {
        if (entry.is_regular_file() && hasBinExtension(entry.path())) {
            lidar_files.push_back(entry.path());
        }
    }

    std::sort(lidar_files.begin(), lidar_files.end());

    for (const auto& lidar_path : lidar_files) {
        const std::string stem = lidar_path.stem().string();
        const std::filesystem::path label_path = label_dir / (stem + ".bin");
        if (!std::filesystem::exists(label_path)) {
            continue;
        }

        pairs.push_back(ScanLabelPair{
            stem,
            lidar_path.string(),
            label_path.string(),
        });
    }

    if (pairs.empty()) {
        size_t label_bin_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(label_dir)) {
            if (entry.is_regular_file() && hasBinExtension(entry.path())) {
                ++label_bin_count;
            }
        }

        error_msg = "No matching lidar/label .bin pairs found. lidar_dir=" + lidar_dir.string() +
                    " (bins=" + std::to_string(lidar_files.size()) + "), label_dir=" + label_dir.string() +
                    " (bins=" + std::to_string(label_bin_count) + ").";
        return false;
    }

    return true;
}

bool getScanLabelPair(const std::vector<ScanLabelPair>& pairs, size_t index, ScanLabelPair& pair, std::string& error_msg) {
    if (pairs.empty()) {
        error_msg = "No scan/label pairs available.";
        return false;
    }
    if (index >= pairs.size()) {
        error_msg = "Requested index " + std::to_string(index) + " out of range [0, " +
                    std::to_string(pairs.size() - 1) + "].";
        return false;
    }

    pair = pairs[index];
    return true;
}

}  // namespace continuous_bki

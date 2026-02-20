#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace continuous_bki {

struct DatasetConfig {
    std::string dataset_name;
    std::string dataset_root_path;
    std::string sequence;
    int skip_frames = 0;
    std::string lidar_dir;
    std::string label_dir;
    std::string pose_path;
    std::string calibration_path;
};

struct ScanLabelPair {
    std::string scan_id;
    std::string scan_path;
    std::string label_path;
};

bool loadDatasetConfig(const std::string& config_path, DatasetConfig& config, std::string& error_msg);
bool collectScanLabelPairs(const DatasetConfig& config, std::vector<ScanLabelPair>& pairs, std::string& error_msg);
bool getScanLabelPair(const std::vector<ScanLabelPair>& pairs, size_t index, ScanLabelPair& pair, std::string& error_msg);

}  // namespace continuous_bki

#pragma once

#include <cstdint>
#include <string>
#include <array>
#include <vector>

#include "continuous_bki.hpp"

namespace continuous_bki {

struct PoseRecord {
    int scan_id = -1;
    double timestamp = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
};

struct Transform4x4 {
    // Row-major 4x4 matrix
    std::array<double, 16> m{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
};

// Reads MCD-style scan .bin files with layout [x, y, z, intensity] float32.
// The intensity channel is skipped, and xyz are written to points.
bool readPointCloudBin(const std::string& scan_bin_path, std::vector<Point3D>& points);

// Reads label .bin files with layout [label0, label1, ...] uint32.
bool readLabelBin(const std::string& label_bin_path, std::vector<uint32_t>& labels);

// Reads MCD pose CSV rows: num,timestamp,x,y,z,qx,qy,qz,qw.
// Also supports whitespace-delimited rows with the same value order.
bool readPosesCSV(const std::string& pose_csv_path, std::vector<PoseRecord>& poses);

// Reads body/os_sensor/T from hhs_calib.yaml.
bool readBodyToLidarCalibration(const std::string& calib_yaml_path, Transform4x4& body_to_lidar);

}  // namespace continuous_bki

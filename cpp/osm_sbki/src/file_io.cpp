#include "file_io.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace continuous_bki {

namespace {

bool parsePoseLine(const std::string& line, PoseRecord& pose) {
    if (line.empty() || line[0] == '#') {
        return false;
    }

    std::vector<double> values;
    std::string token;

    if (line.find(',') != std::string::npos) {
        std::stringstream ss(line);
        while (std::getline(ss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                // Ignore non-numeric tokens (e.g. header).
            }
        }
    } else {
        std::stringstream ss(line);
        double value = 0.0;
        while (ss >> value) {
            values.push_back(value);
        }
    }

    if (values.size() < 8) {
        return false;
    }

    pose.scan_id = static_cast<int>(values[0]);
    pose.timestamp = (values.size() > 1) ? values[1] : 0.0;
    pose.x = values[2];
    pose.y = values[3];
    pose.z = values[4];
    pose.qx = values[5];
    pose.qy = values[6];
    pose.qz = values[7];
    pose.qw = (values.size() > 8) ? values[8] : 1.0;
    return true;
}

std::string trim(const std::string& s) {
    const size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool parseMatrixRow(const std::string& line, double row[4]) {
    const size_t lb = line.find('[');
    const size_t rb = line.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) {
        return false;
    }

    std::string inside = line.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(inside);
    std::string token;
    int idx = 0;
    while (std::getline(ss, token, ',')) {
        if (idx >= 4) return false;
        try {
            row[idx++] = std::stod(trim(token));
        } catch (...) {
            return false;
        }
    }
    return idx == 4;
}

}  // namespace

bool readPointCloudBin(const std::string& scan_bin_path, std::vector<Point3D>& points) {
    std::ifstream file(scan_bin_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || (size_bytes % static_cast<std::streamsize>(sizeof(float) * 4)) != 0) {
        return false;
    }

    const size_t num_points =
        static_cast<size_t>(size_bytes) / static_cast<size_t>(sizeof(float) * 4);

    file.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<size_t>(num_points * 4));
    if (!file.read(reinterpret_cast<char*>(raw.data()), size_bytes)) {
        return false;
    }

    points.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        const size_t base = i * 4;
        points[i] = Point3D(raw[base], raw[base + 1], raw[base + 2]);
    }
    return true;
}

bool readLabelBin(const std::string& label_bin_path, std::vector<uint32_t>& labels) {
    std::ifstream file(label_bin_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    const std::streamsize size_bytes = file.tellg();
    if (size_bytes < 0 || (size_bytes % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) {
        return false;
    }

    const size_t num_labels = static_cast<size_t>(size_bytes) / sizeof(uint32_t);
    labels.resize(num_labels);

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(labels.data()), size_bytes)) {
        return false;
    }

    return true;
}

bool readPosesCSV(const std::string& pose_csv_path, std::vector<PoseRecord>& poses) {
    std::ifstream file(pose_csv_path);
    if (!file.is_open()) {
        return false;
    }

    poses.clear();
    std::string line;
    while (std::getline(file, line)) {
        PoseRecord pose;
        if (parsePoseLine(line, pose)) {
            poses.push_back(pose);
        }
    }

    return !poses.empty();
}

bool readBodyToLidarCalibration(const std::string& calib_yaml_path, Transform4x4& body_to_lidar) {
    std::ifstream file(calib_yaml_path);
    if (!file.is_open()) {
        return false;
    }

    bool in_body = false;
    bool in_os_sensor = false;
    bool in_t_block = false;
    int rows_read = 0;

    std::string line;
    while (std::getline(file, line)) {
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;
        }

        if (t == "body:") {
            in_body = true;
            in_os_sensor = false;
            in_t_block = false;
            continue;
        }
        if (!in_body) {
            continue;
        }
        if (t == "os_sensor:") {
            in_os_sensor = true;
            in_t_block = false;
            continue;
        }
        if (in_os_sensor && t == "T:") {
            in_t_block = true;
            rows_read = 0;
            continue;
        }

        if (in_t_block && t.rfind("- [", 0) == 0) {
            if (rows_read >= 4) {
                return false;
            }
            double row[4] = {0.0, 0.0, 0.0, 0.0};
            if (!parseMatrixRow(t, row)) {
                return false;
            }
            for (int j = 0; j < 4; ++j) {
                body_to_lidar.m[static_cast<size_t>(rows_read * 4 + j)] = row[j];
            }
            ++rows_read;
            if (rows_read == 4) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace continuous_bki

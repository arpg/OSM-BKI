// =======================================
// continuous_bki.hpp
// - Chunked sparse hash grid backend (voxel blocks)
// - Dirichlet-style alpha per voxel
// - OSM prior seeding uses confusion-matrix mapping: p_pred = M @ m
// - Optional OSM fallback during inference when voxel has no support
// =======================================

#ifndef CONTINUOUS_BKI_HPP
#define CONTINUOUS_BKI_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace continuous_bki {

// --- Basic Types ---
struct Point3D {
    float x, y, z;
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float dist_sq(const Point3D& o) const {
        float dx = x - o.x, dy = y - o.y, dz = z - o.z;
        return dx*dx + dy*dy + dz*dz;
    }
};

struct Point2D {
    float x, y;
    Point2D() : x(0), y(0) {}
    Point2D(float x_, float y_) : x(x_), y(y_) {}
};

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
};

struct BlockKey {
    int x, y, z;
    bool operator==(const BlockKey& o) const { return x==o.x && y==o.y && z==o.z; }
    bool operator<(const BlockKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

struct BlockKeyHasher {
    std::size_t operator()(const BlockKey& k) const {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
    }
};

// --- OSM & Config Types ---
struct Polygon {
    std::vector<Point2D> points;
    float min_x, max_x, min_y, max_y;
    bool contains(const Point2D& p) const;
    float distance(const Point2D& p) const;
    void computeBounds();
};

struct OSMData {
    std::map<int, std::vector<Polygon>> geometries; // class_idx -> polygons
    std::map<int, std::vector<Point2D>> point_features; // class_idx -> point locations
};

struct Config {
    std::map<int, std::string> labels;
    std::vector<std::vector<float>> confusion_matrix; // [K_pred x K_prior]
    std::map<int, int> label_to_matrix_idx;
    std::map<std::string, int> osm_class_map;
    std::map<int, float> height_filter_map;
    std::vector<std::string> osm_categories;
    std::map<int, int> raw_to_dense;
    std::map<int, int> dense_to_raw;
    int num_total_classes = 0; // assumed to match K_pred for best results

    // OSM XML projection parameters (for aligning lat/lon to lidar world frame)
    bool has_osm_origin = false;
    double osm_origin_lat = 0.0;   // GPS reference latitude
    double osm_origin_lon = 0.0;   // GPS reference longitude
    double osm_world_offset_x = 0.0;  // World-frame X of the GPS reference
    double osm_world_offset_y = 0.0;  // World-frame Y of the GPS reference
};

// --- Block storage ---
struct Block {
    // alpha stored as flat array: (((lx*B + ly)*B + lz)*K + c)
    std::vector<float> alpha;
    int last_updated = 0;
};

// --- Main Class ---
class ContinuousBKI {
public:
    ContinuousBKI(const Config& config,
                  const OSMData& osm_data,
                  float resolution = 0.1f,
                  float l_scale = 0.3f,
                  float sigma_0 = 1.0f,
                  float prior_delta = 5.0f,
                  float height_sigma = 0.3f,
                  bool use_semantic_kernel = true,
                  bool use_spatial_kernel = true,
                  int num_threads = -1,
                  float alpha0 = 1.0f,
                  bool seed_osm_prior = false,
                  float osm_prior_strength = 0.0f,
                  bool osm_fallback_in_infer = true,
                  float lambda_min = 0.8f,
                  float lambda_max = 0.99f);

    // Core Methods
    void update(const std::vector<uint32_t>& labels, const std::vector<Point3D>& points);
    void update(const std::vector<std::vector<float>>& probs, const std::vector<Point3D>& points, const std::vector<float>& weights);

    std::vector<uint32_t> infer(const std::vector<Point3D>& points) const;
    std::vector<std::vector<float>> infer_probs(const std::vector<Point3D>& points) const;

    // Persistence
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // Utilities
    void clear();
    int size() const; // rough allocated voxel count

private:
    static constexpr int BLOCK_SIZE = 8;

    // --- Integer helpers (correct for negatives) ---
    static inline int div_floor(int a, int d) {
        int q = a / d;
        int r = a % d;
        if (r != 0 && ((r > 0) != (d > 0))) q--;
        return q;
    }
    static inline int mod_floor(int a, int d) {
        int m = a - div_floor(a, d) * d;
        if (m < 0) m += d;
        return m;
    }

    // --- Indexing helpers ---
    VoxelKey pointToKey(const Point3D& p) const;
    Point3D keyToPoint(const VoxelKey& k) const;

    BlockKey voxelToBlockKey(const VoxelKey& vk) const;
    void voxelToLocal(const VoxelKey& vk, int& lx, int& ly, int& lz) const;

    inline int flatIndex(int lx, int ly, int lz, int c) const {
        return (((lx * BLOCK_SIZE) + ly) * BLOCK_SIZE + lz) * config_.num_total_classes + c;
    }

    // shard selection by BlockKey
    int getShardIndex(const BlockKey& bk) const;

    // block allocator/accessor (buffer versions avoid heap allocs during seeding)
    Block& getOrCreateBlock(std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map,
                            const BlockKey& bk,
                            std::vector<float>& buf_m_i,
                            std::vector<float>& buf_p_super,
                            std::vector<float>& buf_p_pred,
                            int current_time) const;
    const Block* getBlockConst(const std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map, const BlockKey& bk) const;

    // Prior seeding (Dirichlet base + optional OSM mapped prior)
    void initVoxelAlpha(Block& b, int lx, int ly, int lz, const Point3D& center,
                        std::vector<float>& buf_m_i,
                        std::vector<float>& buf_p_super,
                        std::vector<float>& buf_p_pred) const;

    // OSM->pred probability mapping: p_pred = normalize(M @ m_i)
    void computePredPriorFromOSM(float x, float y,
                                 std::vector<float>& p_pred_out,
                                 std::vector<float>& buf_m_i,
                                 std::vector<float>& buf_p_super) const;

    // --- OSM / kernels ---
    float computeDistanceToClass(float x, float y, int class_idx) const;
    void getOSMPrior(float x, float y, std::vector<float>& m_i) const;
    float getSemanticKernel(int matrix_idx, const std::vector<float>& m_i,
                            std::vector<float>& buf_expected_obs) const;
    float computeSpatialKernel(float dist_sq) const;

    // --- Precomputed OSM prior raster ---
    struct OSMPriorRaster {
        std::vector<float> data;   // [height * width * K_prior] row-major
        int width = 0, height = 0;
        int K_prior = 0;
        float min_x = 0, min_y = 0;
        float cell_size = 1.0f;

        void build(const ContinuousBKI& bki, float res);
        void lookup(float x, float y, std::vector<float>& m_i) const;
    };

    // --- Flat lookup tables for O(1) label mapping ---
    std::vector<int> raw_to_dense_flat_;   // raw_label -> dense, -1 if unmapped
    std::vector<int> dense_to_raw_flat_;   // dense -> raw_label, -1 if unmapped
    std::vector<int> label_to_matrix_flat_; // raw_label -> matrix_idx, -1 if unmapped
    int max_raw_label_ = 0;

private:
    Config config_;
    OSMData osm_data_;

    // Backend: sharded block hash maps
    std::vector<std::unordered_map<BlockKey, Block, BlockKeyHasher>> block_shards_;

    // Params
    float resolution_;
    float l_scale_;
    float sigma_0_;
    float delta_;
    float height_sigma_;
    float epsilon_;
    bool use_semantic_kernel_;
    bool use_spatial_kernel_;
    int num_threads_;

    // Prior controls
    float alpha0_;
    bool seed_osm_prior_;
    float osm_prior_strength_;
    bool osm_fallback_in_infer_;
    float lambda_min_;
    float lambda_max_;
    int current_time_ = 0;

    // Derived
    int K_pred_;
    int K_prior_;

    // Precomputed structures
    OSMPriorRaster osm_prior_raster_;

    // Reverse mapping: confusion-matrix row index -> list of dense class indices
    std::vector<std::vector<int>> matrix_idx_to_dense_;
};

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

#endif // CONTINUOUS_BKI_HPP

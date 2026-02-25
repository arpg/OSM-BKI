#include "continuous_bki.hpp"
#include "yaml_parser.hpp"
#include "osm_xml_parser.hpp"
#include <fstream>
#include <limits>
#include <climits>
#include <cstring>
#include <numeric>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace continuous_bki {

// --- Polygon Implementation ---
void Polygon::computeBounds() {
    if (points.empty()) {
        min_x = max_x = min_y = max_y = 0.0f;
        return;
    }
    min_x = max_x = points[0].x;
    min_y = max_y = points[0].y;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x < min_x) min_x = points[i].x;
        if (points[i].x > max_x) max_x = points[i].x;
        if (points[i].y < min_y) min_y = points[i].y;
        if (points[i].y > max_y) max_y = points[i].y;
    }
}

bool Polygon::contains(const Point2D& p) const {
    if (points.size() < 3) return false;
    if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) return false;

    bool inside = false;
    for (size_t i = 0, j = points.size() - 1; i < points.size(); j = i++) {
        if ((points[i].y > p.y) != (points[j].y > p.y) &&
            p.x < (points[j].x - points[i].x) * (p.y - points[i].y) /
                  (points[j].y - points[i].y) + points[i].x) {
            inside = !inside;
        }
    }
    return inside;
}

float Polygon::distance(const Point2D& p) const {
    if (points.empty()) return std::numeric_limits<float>::max();

    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < points.size(); i++) {
        size_t j = (i + 1) % points.size();
        const Point2D& p1 = points[i];
        const Point2D& p2 = points[j];
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float len_sq = dx*dx + dy*dy;
        if (len_sq < 1e-10f) {
            float d = std::sqrt((p.x - p1.x)*(p.x - p1.x) + (p.y - p1.y)*(p.y - p1.y));
            min_dist = std::min(min_dist, d);
            continue;
        }
        float t = std::max(0.0f, std::min(1.0f, ((p.x - p1.x) * dx + (p.y - p1.y) * dy) / len_sq));
        float proj_x = p1.x + t * dx;
        float proj_y = p1.y + t * dy;
        float d = std::sqrt((p.x - proj_x)*(p.x - proj_x) + (p.y - proj_y)*(p.y - proj_y));
        min_dist = std::min(min_dist, d);
    }
    return min_dist;
}

// =====================================================================
// OSM Prior Raster: precompute 2D prior field for O(1) bilinear lookup
// =====================================================================

// ---------------------------------------------------------------------------
// Free helper: signed distance from (x,y) to the nearest feature of class_idx.
// Negative == inside a polygon.  +50 if no features exist for that class.
// ---------------------------------------------------------------------------

float computeDistanceToClass(const OSMData& osm_data, float x, float y, int class_idx) {
    Point2D p(x, y);
    float min_dist = std::numeric_limits<float>::max();

    auto it_geom = osm_data.geometries.find(class_idx);
    if (it_geom != osm_data.geometries.end()) {
        for (const auto& poly : it_geom->second) {
            float dist_bbox_x = std::max(0.0f, std::max(poly.min_x - p.x, p.x - poly.max_x));
            float dist_bbox_y = std::max(0.0f, std::max(poly.min_y - p.y, p.y - poly.max_y));
            float dist_bbox_sq = dist_bbox_x * dist_bbox_x + dist_bbox_y * dist_bbox_y;
            if (dist_bbox_sq > min_dist * min_dist) continue;

            float dist = poly.distance(p);
            if (poly.contains(p)) dist = -dist;
            min_dist = std::min(min_dist, dist);
        }
    }

    auto it_points = osm_data.point_features.find(class_idx);
    if (it_points != osm_data.point_features.end()) {
        for (const auto& pt : it_points->second) {
            float dx = p.x - pt.x, dy = p.y - pt.y;
            min_dist = std::min(min_dist, std::sqrt(dx*dx + dy*dy));
        }
    }

    return (min_dist == std::numeric_limits<float>::max()) ? 50.0f : min_dist;
}

// ---------------------------------------------------------------------------
// OSMPriorRaster static factory
// ---------------------------------------------------------------------------

std::shared_ptr<OSMPriorRaster> OSMPriorRaster::build(
        const OSMData& osm_data,
        int K_prior_in,
        float delta,
        float epsilon,
        float res)
{
    auto raster = std::make_shared<OSMPriorRaster>();
    raster->K_prior = K_prior_in;

    if (K_prior_in <= 0) {
        return raster;
    }

    float bb_min_x = std::numeric_limits<float>::max();
    float bb_max_x = -std::numeric_limits<float>::max();
    float bb_min_y = std::numeric_limits<float>::max();
    float bb_max_y = -std::numeric_limits<float>::max();
    bool has_data = false;

    for (const auto& kv : osm_data.geometries) {
        for (const auto& poly : kv.second) {
            bb_min_x = std::min(bb_min_x, poly.min_x);
            bb_max_x = std::max(bb_max_x, poly.max_x);
            bb_min_y = std::min(bb_min_y, poly.min_y);
            bb_max_y = std::max(bb_max_y, poly.max_y);
            has_data = true;
        }
    }
    for (const auto& kv : osm_data.point_features) {
        for (const auto& pt : kv.second) {
            bb_min_x = std::min(bb_min_x, pt.x);
            bb_max_x = std::max(bb_max_x, pt.x);
            bb_min_y = std::min(bb_min_y, pt.y);
            bb_max_y = std::max(bb_max_y, pt.y);
            has_data = true;
        }
    }

    if (!has_data) {
        return raster;
    }

    float pad = delta * 6.0f;
    bb_min_x -= pad; bb_max_x += pad;
    bb_min_y -= pad; bb_max_y += pad;

    raster->min_x     = bb_min_x;
    raster->min_y     = bb_min_y;
    raster->cell_size = res;

    float extent_x = bb_max_x - bb_min_x;
    float extent_y = bb_max_y - bb_min_y;
    raster->width  = static_cast<int>(std::ceil(extent_x / res)) + 1;
    raster->height = static_cast<int>(std::ceil(extent_y / res)) + 1;

    const int W = raster->width, H = raster->height, K = K_prior_in;
    raster->data.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(K));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
    #endif
    for (int iy = 0; iy < H; iy++) {
        float y = raster->min_y + (iy + 0.5f) * res;
        for (int ix = 0; ix < W; ix++) {
            float x = raster->min_x + (ix + 0.5f) * res;
            size_t base = (static_cast<size_t>(iy) * static_cast<size_t>(W)
                         + static_cast<size_t>(ix)) * static_cast<size_t>(K);
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float dist  = computeDistanceToClass(osm_data, x, y, k);
                float score = (dist <= 0.0f)   ? 1.0f
                            : (dist >= delta)  ? 0.0f
                            : 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * dist / delta));
                raster->data[base + static_cast<size_t>(k)] = score;
                sum += score;
            }
            if (sum > epsilon) {
                for (int k = 0; k < K; k++)
                    raster->data[base + static_cast<size_t>(k)] /= sum;
            }
        }
    }

    std::cout << "OSM prior raster: " << W << "x" << H
              << " (" << (W * H) << " cells at " << res << "m resolution)" << std::endl;
    return raster;
}

void OSMPriorRaster::lookup(float x, float y, std::vector<float>& m_i) const {
    if (width <= 0 || height <= 0) return;

    float fx = (x - min_x) / cell_size - 0.5f;
    float fy = (y - min_y) / cell_size - 0.5f;

    int ix0_unclamped = static_cast<int>(std::floor(fx));
    int iy0_unclamped = static_cast<int>(std::floor(fy));
    float tx = fx - static_cast<float>(ix0_unclamped);
    float ty = fy - static_cast<float>(iy0_unclamped);

    int ix0 = std::max(0, std::min(ix0_unclamped, width - 1));
    int iy0 = std::max(0, std::min(iy0_unclamped, height - 1));
    int ix1 = std::max(0, std::min(ix0_unclamped + 1, width - 1));
    int iy1 = std::max(0, std::min(iy0_unclamped + 1, height - 1));

    size_t base00 = (static_cast<size_t>(iy0) * static_cast<size_t>(width) + static_cast<size_t>(ix0)) * static_cast<size_t>(K_prior);
    size_t base10 = (static_cast<size_t>(iy0) * static_cast<size_t>(width) + static_cast<size_t>(ix1)) * static_cast<size_t>(K_prior);
    size_t base01 = (static_cast<size_t>(iy1) * static_cast<size_t>(width) + static_cast<size_t>(ix0)) * static_cast<size_t>(K_prior);
    size_t base11 = (static_cast<size_t>(iy1) * static_cast<size_t>(width) + static_cast<size_t>(ix1)) * static_cast<size_t>(K_prior);

    float w00 = (1.0f - tx) * (1.0f - ty);
    float w10 = tx * (1.0f - ty);
    float w01 = (1.0f - tx) * ty;
    float w11 = tx * ty;

    for (int k = 0; k < K_prior; k++) {
        size_t sk = static_cast<size_t>(k);
        m_i[sk] = w00 * data[base00 + sk] + w10 * data[base10 + sk]
                + w01 * data[base01 + sk] + w11 * data[base11 + sk];
    }
}

// =====================================================================
// ContinuousBKI Implementation
// =====================================================================

ContinuousBKI::ContinuousBKI(const Config& config,
              const OSMData& osm_data,
              float resolution,
              float l_scale,
              float sigma_0,
              float prior_delta,
              float height_sigma,
              bool use_semantic_kernel,
              bool use_spatial_kernel,
              int num_threads,
              float alpha0,
              bool seed_osm_prior,
              float osm_prior_strength,
              bool osm_fallback_in_infer,
              std::shared_ptr<const OSMPriorRaster> shared_raster)
    : config_(config),
      osm_data_(osm_data),
      resolution_(resolution),
      l_scale_(l_scale),
      sigma_0_(sigma_0),
      delta_(prior_delta),
      height_sigma_(height_sigma),
      epsilon_(1e-6f),
      use_semantic_kernel_(use_semantic_kernel),
      use_spatial_kernel_(use_spatial_kernel),
      num_threads_(num_threads),
      alpha0_(alpha0),
      seed_osm_prior_(seed_osm_prior),
      osm_prior_strength_(osm_prior_strength),
      osm_fallback_in_infer_(osm_fallback_in_infer),
      current_time_(0)
{
    K_pred_ = config.confusion_matrix.size();
    K_prior_ = config.confusion_matrix.empty() ? 0 : static_cast<int>(config.confusion_matrix[0].size());

    confusion_matrix_.resize(K_pred_, K_prior_);
    for (int i = 0; i < K_pred_; i++)
        for (int j = 0; j < K_prior_; j++)
            confusion_matrix_(i, j) = config.confusion_matrix[i][j];

#ifdef _OPENMP
    if (num_threads_ < 0) {
        num_threads_ = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads_);
#else
    num_threads_ = 1;
#endif

    block_shards_.resize(static_cast<size_t>(num_threads_));

    // Build flat lookup tables for O(1) label mapping.
    // dense index == confusion-matrix row (labels are 0..N_COMMON-1).
    max_raw_label_ = 0;
    for (const auto& kv : config_.raw_to_dense) {
        if (kv.first > max_raw_label_) max_raw_label_ = kv.first;
    }

    raw_to_dense_flat_.assign(static_cast<size_t>(max_raw_label_ + 1), -1);
    for (const auto& kv : config_.raw_to_dense) {
        raw_to_dense_flat_[static_cast<size_t>(kv.first)] = kv.second;
    }

    int max_dense = config_.num_total_classes;
    dense_to_raw_flat_.assign(static_cast<size_t>(max_dense), -1);
    for (const auto& kv : config_.dense_to_raw) {
        if (kv.first >= 0 && kv.first < max_dense) {
            dense_to_raw_flat_[static_cast<size_t>(kv.first)] = kv.second;
        }
    }

    inv_l_scale_sq_ = 1.0f / (l_scale_ * l_scale_);
    spatial_kernel_lut_.resize(SPATIAL_KERNEL_LUT_SIZE + 1);
    for (int i = 0; i <= SPATIAL_KERNEL_LUT_SIZE; i++) {
        float t = static_cast<float>(i) / static_cast<float>(SPATIAL_KERNEL_LUT_SIZE);
        float xi = std::sqrt(t);
        if (xi < 1.0f) {
            float term1 = (1.0f / 3.0f) * (2.0f + std::cos(2.0f * static_cast<float>(M_PI) * xi)) * (1.0f - xi);
            float term2 = (1.0f / (2.0f * static_cast<float>(M_PI))) * std::sin(2.0f * static_cast<float>(M_PI) * xi);
            spatial_kernel_lut_[i] = sigma_0_ * (term1 + term2);
        } else {
            spatial_kernel_lut_[i] = 0.0f;
        }
    }

    if (shared_raster) {
        osm_prior_raster_ = shared_raster;
        std::cout << "[Profiling] OSM raster build: 0 ms (reusing shared raster)" << std::endl;
    } else {
        auto t0_raster = std::chrono::high_resolution_clock::now();
        osm_prior_raster_ = OSMPriorRaster::build(osm_data_, K_prior_, delta_, epsilon_, resolution_);
        auto t1_raster = std::chrono::high_resolution_clock::now();
        profiling_.raster_build_ms = std::chrono::duration<double, std::milli>(t1_raster - t0_raster).count();
        std::cout << "[Profiling] OSM raster build: " << profiling_.raster_build_ms << " ms" << std::endl;
    }
}

void ContinuousBKI::clear() {
    for (auto& shard : block_shards_) {
        shard.clear();
    }
}

int ContinuousBKI::getShardIndex(const BlockKey& k) const {
    BlockKeyHasher hasher;
    return static_cast<int>(hasher(k) % block_shards_.size());
}

VoxelKey ContinuousBKI::pointToKey(const Point3D& p) const {
    return VoxelKey{
        static_cast<int>(std::floor(p.x / resolution_)),
        static_cast<int>(std::floor(p.y / resolution_)),
        static_cast<int>(std::floor(p.z / resolution_))
    };
}

Point3D ContinuousBKI::keyToPoint(const VoxelKey& k) const {
    return Point3D(
        (k.x + 0.5f) * resolution_,
        (k.y + 0.5f) * resolution_,
        (k.z + 0.5f) * resolution_
    );
}

BlockKey ContinuousBKI::voxelToBlockKey(const VoxelKey& vk) const {
    return BlockKey{
        div_floor(vk.x, BLOCK_SIZE),
        div_floor(vk.y, BLOCK_SIZE),
        div_floor(vk.z, BLOCK_SIZE)
    };
}

void ContinuousBKI::voxelToLocal(const VoxelKey& vk, int& lx, int& ly, int& lz) const {
    lx = mod_floor(vk.x, BLOCK_SIZE);
    ly = mod_floor(vk.y, BLOCK_SIZE);
    lz = mod_floor(vk.z, BLOCK_SIZE);
}

// getOrCreateBlock with pre-allocated buffers to avoid heap allocs during OSM seeding
Block& ContinuousBKI::getOrCreateBlock(
        BlockMap& shard_map,
        const BlockKey& bk,
        std::vector<float>& buf_m_i,
        std::vector<float>& buf_p_super,
        std::vector<float>& buf_p_pred) const {
    auto it = shard_map.find(bk);
    if (it != shard_map.end()) {
        return it->second;
    }

    profiling_.new_block_count.fetch_add(1, std::memory_order_relaxed);

    Block blk;
    const size_t total = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);
    blk.alpha.resize(total, alpha0_);
    blk.last_updated = current_time_;

    if (seed_osm_prior_ && osm_prior_strength_ > 0.0f) {
        for (int lz = 0; lz < BLOCK_SIZE; lz++) {
            for (int ly = 0; ly < BLOCK_SIZE; ly++) {
                for (int lx = 0; lx < BLOCK_SIZE; lx++) {
                    int vx = bk.x * BLOCK_SIZE + lx;
                    int vy = bk.y * BLOCK_SIZE + ly;
                    int vz = bk.z * BLOCK_SIZE + lz;
                    Point3D center((vx + 0.5f) * resolution_, (vy + 0.5f) * resolution_, (vz + 0.5f) * resolution_);
                    initVoxelAlpha(blk, lx, ly, lz, center, buf_m_i, buf_p_super, buf_p_pred);
                }
            }
        }
    }

    auto inserted = shard_map.emplace(bk, std::move(blk));
    return inserted.first->second;
}

const Block* ContinuousBKI::getBlockConst(const BlockMap& shard_map, const BlockKey& bk) const {
    auto it = shard_map.find(bk);
    if (it == shard_map.end()) return nullptr;
    return &it->second;
}

// initVoxelAlpha with pre-allocated buffers
void ContinuousBKI::initVoxelAlpha(Block& b, int lx, int ly, int lz, const Point3D& center,
                                    std::vector<float>& buf_m_i,
                                    std::vector<float>& buf_p_super,
                                    std::vector<float>& buf_p_pred) const {
    const int K = config_.num_total_classes;
    for (int c = 0; c < K; c++) {
        int idx = flatIndex(lx, ly, lz, c);
        b.alpha[idx] = alpha0_;
    }
    if (seed_osm_prior_ && osm_prior_strength_ > 0.0f && K_pred_ > 0 && K_prior_ > 0) {
        computePredPriorFromOSM(center.x, center.y, buf_p_pred, buf_m_i, buf_p_super);
        if (buf_p_pred.size() == static_cast<size_t>(K)) {
            for (int c = 0; c < K; c++) {
                int idx = flatIndex(lx, ly, lz, c);
                b.alpha[idx] += osm_prior_strength_ * buf_p_pred[c];
            }
        }
    }
}

void ContinuousBKI::computePredPriorFromOSM(float x, float y,
                                              std::vector<float>& p_pred_out,
                                              std::vector<float>& buf_m_i,
                                              std::vector<float>& buf_p_super) const {
    const int K = config_.num_total_classes;

    if (buf_m_i.size() != static_cast<size_t>(K_prior_))
        buf_m_i.resize(static_cast<size_t>(K_prior_));
    getOSMPrior(x, y, buf_m_i);

    if (buf_p_super.size() != static_cast<size_t>(K_pred_))
        buf_p_super.resize(static_cast<size_t>(K_pred_));

    Eigen::Map<Eigen::VectorXf> m_i_vec(buf_m_i.data(), K_prior_);
    Eigen::Map<Eigen::VectorXf> p_super_vec(buf_p_super.data(), K_pred_);
    p_super_vec.noalias() = confusion_matrix_ * m_i_vec;

    // Dense index == matrix row, so prior maps directly to p_pred.
    if (p_pred_out.size() != static_cast<size_t>(K))
        p_pred_out.resize(static_cast<size_t>(K));
    std::fill(p_pred_out.begin(), p_pred_out.end(), 0.0f);
    for (int i = 0; i < K_pred_ && i < K; i++) {
        p_pred_out[static_cast<size_t>(i)] = buf_p_super[i];
    }

    Eigen::Map<Eigen::VectorXf> p_pred_vec(p_pred_out.data(), K);
    float sum = p_pred_vec.sum();
    if (sum > epsilon_) p_pred_vec /= sum;
}

float ContinuousBKI::computeSpatialKernel(float dist_sq) const {
    if (!use_spatial_kernel_) return 1.0f;

    float t = dist_sq * inv_l_scale_sq_;
    if (t >= 1.0f) return 0.0f;

    float fidx = t * static_cast<float>(SPATIAL_KERNEL_LUT_SIZE);
    int idx = static_cast<int>(fidx);
    float frac = fidx - static_cast<float>(idx);
    return spatial_kernel_lut_[idx] + frac * (spatial_kernel_lut_[idx + 1] - spatial_kernel_lut_[idx]);
}

float ContinuousBKI::computeDistanceToClass(float x, float y, int class_idx) const {
    return continuous_bki::computeDistanceToClass(osm_data_, x, y, class_idx);
}

// getOSMPrior: uses precomputed raster when available, falls back to brute force
void ContinuousBKI::getOSMPrior(float x, float y, std::vector<float>& m_i) const {
    if (m_i.size() != static_cast<size_t>(K_prior_)) m_i.resize(static_cast<size_t>(K_prior_));

    if (osm_prior_raster_ && osm_prior_raster_->width > 0) {
        osm_prior_raster_->lookup(x, y, m_i);
        return;
    }

    // Fallback: compute on the fly (no raster built)
    for (int k = 0; k < K_prior_; k++) {
        float dist = computeDistanceToClass(x, y, k);
        m_i[k] = (dist <= 0.0f) ? 1.0f
               : (dist >= delta_) ? 0.0f
               : 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * dist / delta_));
    }
    Eigen::Map<Eigen::VectorXf> m_i_vec(m_i.data(), K_prior_);
    float sum = m_i_vec.sum();
    if (sum > epsilon_) m_i_vec /= sum;
}

float ContinuousBKI::getSemanticKernel(int matrix_idx, const std::vector<float>& m_i,
                                        std::vector<float>& buf_expected_obs) const {
    if (!use_semantic_kernel_) return 1.0f;
    if (matrix_idx < 0 || matrix_idx >= K_pred_) return 1.0f;

    Eigen::Map<const Eigen::VectorXf> m_i_vec(m_i.data(), K_prior_);
    float c_xi = m_i_vec.maxCoeff();

    if (buf_expected_obs.size() != static_cast<size_t>(K_pred_))
        buf_expected_obs.resize(static_cast<size_t>(K_pred_));

    Eigen::Map<Eigen::VectorXf> obs_vec(buf_expected_obs.data(), K_pred_);
    obs_vec.noalias() = confusion_matrix_ * m_i_vec;

    float numerator = obs_vec(matrix_idx);
    float denominator = obs_vec.maxCoeff() + epsilon_;
    float s_i = numerator / denominator;
    return (1.0f - c_xi) + (c_xi * s_i);
}

// =====================================================================
// update (labels overload)
// =====================================================================
void ContinuousBKI::update(const std::vector<uint32_t>& labels, const std::vector<Point3D>& points) {
    update_impl(labels, points, {}, false);
}

// =====================================================================
// update (probs overload)
// =====================================================================
void ContinuousBKI::update(const std::vector<std::vector<float>>& probs, const std::vector<Point3D>& points, const std::vector<float>& weights) {
    update_impl(probs, points, weights, !weights.empty());
}

template <typename ValueType>
void ContinuousBKI::update_impl(const std::vector<ValueType>& values,
                                const std::vector<Point3D>& points,
                                const std::vector<float>& weights,
                                bool use_weights) {
    if (values.size() != points.size()) {
        std::cerr << "Mismatch in points and values size" << std::endl;
        return;
    }
    if (use_weights && weights.size() != points.size()) {
        std::cerr << "Mismatch in points and weights size" << std::endl;
        return;
    }

    size_t n = points.size();
    auto t_update_start = std::chrono::high_resolution_clock::now();

    // Semantic Kernel Pre-computation (only for hard labels)
    std::vector<float> point_k_sem;
    if constexpr (std::is_same<ValueType, uint32_t>::value) {
        point_k_sem.assign(n, 1.0f);
        if (use_semantic_kernel_) {
            #ifdef _OPENMP
            #pragma omp parallel
            {
                std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
                std::vector<float> tl_expected_obs(static_cast<size_t>(K_pred_));
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; i++) {
                    getOSMPrior(points[i].x, points[i].y, tl_m_i);
                    int raw_label = static_cast<int>(values[i]);
                    int matrix_idx = (raw_label >= 0 && raw_label <= max_raw_label_)
                                     ? raw_to_dense_flat_[static_cast<size_t>(raw_label)] : -1;
                    if (matrix_idx >= 0) {
                        point_k_sem[i] = getSemanticKernel(matrix_idx, tl_m_i, tl_expected_obs);
                    }
                }
            }
            #else
            std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
            std::vector<float> tl_expected_obs(static_cast<size_t>(K_pred_));
            for (size_t i = 0; i < n; i++) {
                getOSMPrior(points[i].x, points[i].y, tl_m_i);
                int raw_label = static_cast<int>(values[i]);
                int matrix_idx = (raw_label >= 0 && raw_label <= max_raw_label_)
                                 ? raw_to_dense_flat_[static_cast<size_t>(raw_label)] : -1;
                if (matrix_idx >= 0) {
                    point_k_sem[i] = getSemanticKernel(matrix_idx, tl_m_i, tl_expected_obs);
                }
            }
            #endif
        }
    }

    auto t_sem_done = std::chrono::high_resolution_clock::now();

    current_time_++;
    int num_shards = static_cast<int>(block_shards_.size());
    int radius = static_cast<int>(std::ceil(l_scale_ / resolution_));
    float l_scale_sq = l_scale_ * l_scale_;

    // Precompute spherical neighborhood offsets (avoids per-point sqrt in loop bounds)
    struct VoxelOffset { int dx, dy, dz; };
    std::vector<VoxelOffset> neighbor_offsets;
    {
        int r2 = radius * radius;
        for (int dx = -radius; dx <= radius; dx++)
            for (int dy = -radius; dy <= radius; dy++)
                for (int dz = -radius; dz <= radius; dz++)
                    if (dx * dx + dy * dy + dz * dz <= r2)
                        neighbor_offsets.push_back({dx, dy, dz});
    }

    // Shard assignment
    std::vector<std::vector<size_t>> shard_points(static_cast<size_t>(num_shards));
    for (size_t i = 0; i < n; i++) {
        VoxelKey vk_p = pointToKey(points[i]);
        BlockKey min_bk = voxelToBlockKey({vk_p.x - radius, vk_p.y - radius, vk_p.z - radius});
        BlockKey max_bk = voxelToBlockKey({vk_p.x + radius, vk_p.y + radius, vk_p.z + radius});
        std::vector<bool> added(static_cast<size_t>(num_shards), false);
        for (int bx = min_bk.x; bx <= max_bk.x; bx++) {
            for (int by = min_bk.y; by <= max_bk.y; by++) {
                for (int bz = min_bk.z; bz <= max_bk.z; bz++) {
                    int s = getShardIndex({bx, by, bz});
                    if (!added[static_cast<size_t>(s)]) {
                        shard_points[static_cast<size_t>(s)].push_back(i);
                        added[static_cast<size_t>(s)] = true;
                    }
                }
            }
        }
    }

    auto t_shard_done = std::chrono::high_resolution_clock::now();

    #ifdef _OPENMP
    #pragma omp parallel num_threads(num_shards)
    {
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
        std::vector<float> tl_p_pred(static_cast<size_t>(config_.num_total_classes));
        #pragma omp for schedule(static, 1)
    #endif
    for (int s = 0; s < num_shards; ++s) {
        #ifndef _OPENMP
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
        std::vector<float> tl_p_pred(static_cast<size_t>(config_.num_total_classes));
        #endif
        
        auto& shard = block_shards_[static_cast<size_t>(s)];
        const std::vector<size_t>& pts = shard_points[static_cast<size_t>(s)];

        for (size_t pt_idx = 0; pt_idx < pts.size(); pt_idx++) {
            size_t i = pts[pt_idx];
            const Point3D& p = points[i];
            VoxelKey vk_p = pointToKey(p);
            float w_i = use_weights ? weights[i] : 1.0f;

            // Prepare update value
            int dense_label = -1;
            float k_sem = 1.0f;
            const std::vector<float>* prob_ptr = nullptr;

            if constexpr (std::is_same<ValueType, uint32_t>::value) {
                int raw_label = static_cast<int>(values[i]);
                dense_label = (raw_label >= 0 && raw_label <= max_raw_label_)
                              ? raw_to_dense_flat_[static_cast<size_t>(raw_label)] : -1;
                if (dense_label < 0) continue;
                k_sem = point_k_sem[i];
            } else {
                prob_ptr = &values[i];
            }

            BlockKey cached_bk = {INT_MIN, INT_MIN, INT_MIN};
            Block* cached_blk = nullptr;

            for (const auto& off : neighbor_offsets) {
                VoxelKey vk = {vk_p.x + off.dx, vk_p.y + off.dy, vk_p.z + off.dz};
                BlockKey bk = voxelToBlockKey(vk);
                if (getShardIndex(bk) != s) continue;

                Point3D v_center = keyToPoint(vk);
                float dist_sq = p.dist_sq(v_center);
                if (dist_sq > l_scale_sq) continue;

                float k_sp = computeSpatialKernel(dist_sq);
                if (k_sp <= 1e-6f) continue;

                if (!(bk == cached_bk)) {
                    cached_blk = &getOrCreateBlock(shard, bk, tl_m_i, tl_p_super, tl_p_pred);
                    cached_bk = bk;
                }
                Block& blk = *cached_blk;
                int lx, ly, lz_local;
                voxelToLocal(vk, lx, ly, lz_local);

                if constexpr (std::is_same<ValueType, uint32_t>::value) {
                    int idx = flatIndex(lx, ly, lz_local, dense_label);
                    blk.alpha[static_cast<size_t>(idx)] += k_sp * k_sem;
                } else {
                    size_t K_clamp = std::min(prob_ptr->size(), static_cast<size_t>(config_.num_total_classes));
                    for (size_t c = 0; c < K_clamp; c++) {
                        int idx = flatIndex(lx, ly, lz_local, static_cast<int>(c));
                        blk.alpha[static_cast<size_t>(idx)] += w_i * k_sp * (*prob_ptr)[c];
                    }
                }
            }
        }
    }
    #ifdef _OPENMP
    }
    #endif

    auto t_update_end = std::chrono::high_resolution_clock::now();
    double ms_sem = std::chrono::duration<double, std::milli>(t_sem_done - t_update_start).count();
    double ms_shard = std::chrono::duration<double, std::milli>(t_shard_done - t_sem_done).count();
    double ms_kernel = std::chrono::duration<double, std::milli>(t_update_end - t_shard_done).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_update_end - t_update_start).count();

    profiling_.update_calls++;
    profiling_.total_update_ms += ms_total;
    profiling_.total_semantic_precomp_ms += ms_sem;
    profiling_.total_shard_assign_ms += ms_shard;
    profiling_.total_kernel_update_ms += ms_kernel;

    std::cout << "[Profiling] update #" << profiling_.update_calls
              << ": " << ms_total << " ms total"
              << " (sem_precomp=" << ms_sem << " ms"
              << ", shard_assign=" << ms_shard << " ms"
              << ", kernel_update=" << ms_kernel << " ms)"
              << ", points=" << n
              << ", radius=" << radius
              << ", new_blocks=" << profiling_.new_block_count.load()
              << std::endl;

    profiling_.new_block_count.store(0);
}

int ContinuousBKI::size() const {
    int total = 0;
    for (const auto& shard : block_shards_) {
        total += static_cast<int>(shard.size()) * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    }
    return total;
}

void ContinuousBKI::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving: " << filename << std::endl;
        return;
    }

    const uint8_t version = 3;
    out.write(reinterpret_cast<const char*>(&version), sizeof(uint8_t));
    out.write(reinterpret_cast<const char*>(&resolution_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&l_scale_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&sigma_0_), sizeof(float));
    out.write(reinterpret_cast<const char*>(&current_time_), sizeof(int));

    size_t num_blocks = 0;
    for (const auto& shard : block_shards_) num_blocks += shard.size();
    out.write(reinterpret_cast<const char*>(&num_blocks), sizeof(size_t));

    const size_t block_alpha_size = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);
    for (const auto& shard : block_shards_) {
        for (const auto& kv : shard) {
            const BlockKey& bk = kv.first;
            const Block& blk = kv.second;
            out.write(reinterpret_cast<const char*>(&bk.x), sizeof(int));
            out.write(reinterpret_cast<const char*>(&bk.y), sizeof(int));
            out.write(reinterpret_cast<const char*>(&bk.z), sizeof(int));
            out.write(reinterpret_cast<const char*>(&blk.last_updated), sizeof(int));
            if (blk.alpha.size() == block_alpha_size) {
                out.write(reinterpret_cast<const char*>(blk.alpha.data()), block_alpha_size * sizeof(float));
            }
        }
    }
    out.close();
}

void ContinuousBKI::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading: " << filename << std::endl;
        return;
    }

    uint8_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    if (version != 2 && version != 3) {
        std::cerr << "Unsupported map file version: " << static_cast<int>(version) << " (expected 2 or 3)" << std::endl;
        return;
    }

    float res, l, s0;
    in.read(reinterpret_cast<char*>(&res), sizeof(float));
    in.read(reinterpret_cast<char*>(&l), sizeof(float));
    in.read(reinterpret_cast<char*>(&s0), sizeof(float));

    if (version >= 3) {
        in.read(reinterpret_cast<char*>(&current_time_), sizeof(int));
    } else {
        current_time_ = 0;
    }

    if (std::abs(res - resolution_) > 1e-4f) std::cerr << "Warning: Loaded resolution mismatch" << std::endl;

    size_t num_blocks = 0;
    in.read(reinterpret_cast<char*>(&num_blocks), sizeof(size_t));

    clear();
    const size_t block_alpha_size = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);

    for (size_t i = 0; i < num_blocks; i++) {
        BlockKey bk;
        in.read(reinterpret_cast<char*>(&bk.x), sizeof(int));
        in.read(reinterpret_cast<char*>(&bk.y), sizeof(int));
        in.read(reinterpret_cast<char*>(&bk.z), sizeof(int));
        Block blk;
        blk.alpha.resize(block_alpha_size);
        if (version >= 3) {
            in.read(reinterpret_cast<char*>(&blk.last_updated), sizeof(int));
        } else {
            blk.last_updated = current_time_; // Assume fresh if loading old map
        }
        in.read(reinterpret_cast<char*>(blk.alpha.data()), block_alpha_size * sizeof(float));
        int s = getShardIndex(bk);
        block_shards_[static_cast<size_t>(s)][bk] = std::move(blk);
    }
    in.close();
}

// =====================================================================
// infer - parallelized with OpenMP
// =====================================================================
std::vector<uint32_t> ContinuousBKI::infer(const std::vector<Point3D>& points) const {
    return infer_impl<uint32_t>(points, false);
}

// =====================================================================
// infer_probs - parallelized with OpenMP
// =====================================================================
std::vector<std::vector<float>> ContinuousBKI::infer_probs(const std::vector<Point3D>& points) const {
    return infer_impl<std::vector<float>>(points, true);
}

template <typename ResultType>
std::vector<ResultType> ContinuousBKI::infer_impl(const std::vector<Point3D>& points, bool return_probs) const {
    auto t_infer_start = std::chrono::high_resolution_clock::now();
    const size_t n = points.size();
    std::vector<ResultType> results(n);
    const int K = config_.num_total_classes;

    // Default values
    ResultType default_val;
    if constexpr (std::is_same<ResultType, uint32_t>::value) {
        default_val = 0;
    } else {
        default_val.assign(K, 1.0f / K);
    }

    #ifdef _OPENMP
    #pragma omp parallel
    {
        std::vector<float> tl_p_pred(static_cast<size_t>(K));
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
        std::vector<float> tl_probs; 
        if (return_probs) tl_probs.resize(K);

        #pragma omp for schedule(static)
    #else
        std::vector<float> tl_p_pred(static_cast<size_t>(K));
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
        std::vector<float> tl_probs; 
        if (return_probs) tl_probs.resize(K);
    #endif
        for (size_t i = 0; i < n; i++) {
            const Point3D& p = points[i];
            VoxelKey k = pointToKey(p);
            BlockKey bk = voxelToBlockKey(k);
            int s = getShardIndex(bk);
            const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

            bool found_in_block = false;
            if (blk != nullptr) {
                int lx, ly, lz;
                voxelToLocal(k, lx, ly, lz);
                
                Eigen::Map<const Eigen::VectorXf> alpha_vec(
                    blk->alpha.data() + flatIndex(lx, ly, lz, 0), K);

                if (!return_probs) {
                    float sum = alpha_vec.sum();
                    if (sum > epsilon_) {
                        Eigen::Index best_idx;
                        alpha_vec.maxCoeff(&best_idx);
                        int raw = (best_idx >= 0 && best_idx < static_cast<Eigen::Index>(dense_to_raw_flat_.size()))
                                  ? dense_to_raw_flat_[best_idx] : -1;
                        if constexpr (std::is_same<ResultType, uint32_t>::value) {
                            results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                        }
                        found_in_block = true;
                    }
                } else {
                    float sum = alpha_vec.sum();
                    if (sum > epsilon_) {
                        Eigen::Map<Eigen::VectorXf>(tl_probs.data(), K) = alpha_vec / sum;
                        if constexpr (std::is_same<ResultType, std::vector<float>>::value) {
                            results[i] = tl_probs;
                        }
                        found_in_block = true;
                    }
                }
            }

            if (found_in_block) continue;

            // Fallback to OSM
            if (osm_fallback_in_infer_ && K_pred_ > 0) {
                computePredPriorFromOSM(p.x, p.y, tl_p_pred, tl_m_i, tl_p_super);
                
                if (!return_probs) {
                    if (!tl_p_pred.empty()) {
                        Eigen::Map<Eigen::VectorXf> pred_vec(tl_p_pred.data(), K);
                        Eigen::Index best;
                        pred_vec.maxCoeff(&best);
                        int raw = (best >= 0 && best < static_cast<Eigen::Index>(dense_to_raw_flat_.size()))
                                  ? dense_to_raw_flat_[best] : -1;
                        if constexpr (std::is_same<ResultType, uint32_t>::value) {
                            results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                        }
                    } else {
                        results[i] = default_val;
                    }
                } else {
                    if (tl_p_pred.size() == static_cast<size_t>(K)) {
                        if constexpr (std::is_same<ResultType, std::vector<float>>::value) {
                            results[i] = tl_p_pred;
                        }
                    } else {
                        results[i] = default_val;
                    }
                }
            } else {
                results[i] = default_val;
            }
        }
    #ifdef _OPENMP
    }
    #endif

    auto t_infer_end = std::chrono::high_resolution_clock::now();
    double ms_infer = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();
    profiling_.infer_calls++;
    profiling_.total_infer_ms += ms_infer;

    std::cout << "[Profiling] infer #" << profiling_.infer_calls
              << ": " << ms_infer << " ms, points=" << n << std::endl;

    return results;
}

void ContinuousBKI::printProfilingStats() const {
    std::cout << "\n=== ContinuousBKI Profiling Summary ===" << std::endl;
    std::cout << "  OSM raster build:        " << profiling_.raster_build_ms << " ms (one-time)" << std::endl;
    std::cout << "  Update calls:            " << profiling_.update_calls << std::endl;
    std::cout << "    Total update time:     " << profiling_.total_update_ms << " ms" << std::endl;
    if (profiling_.update_calls > 0) {
        std::cout << "    Avg per update:        " << profiling_.total_update_ms / profiling_.update_calls << " ms" << std::endl;
        std::cout << "    Semantic precomp:      " << profiling_.total_semantic_precomp_ms << " ms ("
                  << (100.0 * profiling_.total_semantic_precomp_ms / profiling_.total_update_ms) << "%)" << std::endl;
        std::cout << "    Shard assignment:      " << profiling_.total_shard_assign_ms << " ms ("
                  << (100.0 * profiling_.total_shard_assign_ms / profiling_.total_update_ms) << "%)" << std::endl;
        std::cout << "    Kernel update:         " << profiling_.total_kernel_update_ms << " ms ("
                  << (100.0 * profiling_.total_kernel_update_ms / profiling_.total_update_ms) << "%)" << std::endl;
    }
    std::cout << "  Infer calls:             " << profiling_.infer_calls << std::endl;
    std::cout << "    Total infer time:      " << profiling_.total_infer_ms << " ms" << std::endl;
    if (profiling_.infer_calls > 0) {
        std::cout << "    Avg per infer:         " << profiling_.total_infer_ms / profiling_.infer_calls << " ms" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

// --- Loader Implementations ---
// Moved to osm_loader.cpp

} // namespace continuous_bki

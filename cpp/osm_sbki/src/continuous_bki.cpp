#include "continuous_bki.hpp"
#include "yaml_parser.hpp"
#include "osm_xml_parser.hpp"
#include <fstream>
#include <limits>
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

// Brute-force distance to class (all features, no spatial index).
// Used only during one-time raster construction.
static float computeDistanceToClassBruteForce(const OSMData& osm_data,
                                               float x, float y, int class_idx) {
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
            float dx = p.x - pt.x;
            float dy = p.y - pt.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            min_dist = std::min(min_dist, dist);
        }
    }

    if (min_dist == std::numeric_limits<float>::max()) return 50.0f;
    return min_dist;
}

void ContinuousBKI::OSMPriorRaster::build(const ContinuousBKI& bki, float res) {
    K_prior = bki.K_prior_;
    if (K_prior <= 0) {
        width = height = 0;
        return;
    }

    // Compute bounding box from all OSM features
    float bb_min_x = std::numeric_limits<float>::max();
    float bb_max_x = -std::numeric_limits<float>::max();
    float bb_min_y = std::numeric_limits<float>::max();
    float bb_max_y = -std::numeric_limits<float>::max();
    bool has_data = false;

    for (const auto& kv : bki.osm_data_.geometries) {
        for (const auto& poly : kv.second) {
            bb_min_x = std::min(bb_min_x, poly.min_x);
            bb_max_x = std::max(bb_max_x, poly.max_x);
            bb_min_y = std::min(bb_min_y, poly.min_y);
            bb_max_y = std::max(bb_max_y, poly.max_y);
            has_data = true;
        }
    }
    for (const auto& kv : bki.osm_data_.point_features) {
        for (const auto& pt : kv.second) {
            bb_min_x = std::min(bb_min_x, pt.x);
            bb_max_x = std::max(bb_max_x, pt.x);
            bb_min_y = std::min(bb_min_y, pt.y);
            bb_max_y = std::max(bb_max_y, pt.y);
            has_data = true;
        }
    }

    if (!has_data) {
        width = height = 0;
        return;
    }

    // Pad by sigmoid effective range so edge queries still get meaningful priors
    float pad = bki.delta_ * 6.0f;
    bb_min_x -= pad; bb_max_x += pad;
    bb_min_y -= pad; bb_max_y += pad;

    min_x = bb_min_x;
    min_y = bb_min_y;
    cell_size = res;

    float extent_x = bb_max_x - bb_min_x;
    float extent_y = bb_max_y - bb_min_y;
    width  = static_cast<int>(std::ceil(extent_x / cell_size)) + 1;
    height = static_cast<int>(std::ceil(extent_y / cell_size)) + 1;

    data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(K_prior));

    for (int iy = 0; iy < height; iy++) {
        float y = min_y + (iy + 0.5f) * cell_size;
        for (int ix = 0; ix < width; ix++) {
            float x = min_x + (ix + 0.5f) * cell_size;
            size_t base = (static_cast<size_t>(iy) * static_cast<size_t>(width)
                         + static_cast<size_t>(ix)) * static_cast<size_t>(K_prior);

            float sum = 0.0f;
            for (int k = 0; k < K_prior; k++) {
                float dist = computeDistanceToClassBruteForce(bki.osm_data_, x, y, k);
                float score = 1.0f / (1.0f + std::exp((dist / bki.delta_) - 4.6f));
                data[base + static_cast<size_t>(k)] = score;
                sum += score;
            }
            if (sum > bki.epsilon_) {
                for (int k = 0; k < K_prior; k++) {
                    data[base + static_cast<size_t>(k)] /= sum;
                }
            }
        }
    }

    std::cout << "OSM prior raster: " << width << "x" << height
              << " (" << (width * height) << " cells at " << cell_size << "m resolution)" << std::endl;
}

void ContinuousBKI::OSMPriorRaster::lookup(float x, float y, std::vector<float>& m_i) const {
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
              float lambda_min,
              float lambda_max)
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
      lambda_min_(lambda_min),
      lambda_max_(lambda_max),
      current_time_(0)
{
    K_pred_ = config.confusion_matrix.size();
    K_prior_ = config.confusion_matrix.empty() ? 0 : static_cast<int>(config.confusion_matrix[0].size());

#ifdef _OPENMP
    if (num_threads_ < 0) {
        num_threads_ = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads_);
#else
    num_threads_ = 1;
#endif

    block_shards_.resize(static_cast<size_t>(num_threads_));

    // Build reverse mapping: confusion-matrix row index -> dense class indices.
    matrix_idx_to_dense_.resize(static_cast<size_t>(K_pred_));
    for (const auto& kv : config_.label_to_matrix_idx) {
        int raw_label = kv.first;
        int matrix_idx = kv.second;
        auto it = config_.raw_to_dense.find(raw_label);
        if (it != config_.raw_to_dense.end() && matrix_idx >= 0 && matrix_idx < K_pred_) {
            matrix_idx_to_dense_[static_cast<size_t>(matrix_idx)].push_back(it->second);
        }
    }

    // Build flat lookup tables for O(1) label mapping
    max_raw_label_ = 0;
    for (const auto& kv : config_.raw_to_dense) {
        if (kv.first > max_raw_label_) max_raw_label_ = kv.first;
    }
    for (const auto& kv : config_.label_to_matrix_idx) {
        if (kv.first > max_raw_label_) max_raw_label_ = static_cast<int>(kv.first);
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

    label_to_matrix_flat_.assign(static_cast<size_t>(max_raw_label_ + 1), -1);
    for (const auto& kv : config_.label_to_matrix_idx) {
        label_to_matrix_flat_[static_cast<size_t>(kv.first)] = kv.second;
    }

    // Build precomputed OSM prior raster (one-time cost)
    osm_prior_raster_.build(*this, resolution_);
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
        std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map,
        const BlockKey& bk,
        std::vector<float>& buf_m_i,
        std::vector<float>& buf_p_super,
        std::vector<float>& buf_p_pred,
        int current_time) const {
    auto it = shard_map.find(bk);
    if (it != shard_map.end()) {
        // Lazy decay: if block hasn't been updated this batch, apply forgetting
        Block& blk = it->second;
        if (blk.last_updated < current_time) {
            int dt = current_time - blk.last_updated;
            blk.last_updated = current_time;

            // If lambda_max >= 1.0, maybe we skip decay? 
            // But user asked for lambda in (0,1). Assuming lambda_max < 1.0 implies decay.
            if (lambda_max_ < 1.0f) {
                const int K = config_.num_total_classes;
                
                // Iterate all voxels in the block
                for (int lz = 0; lz < BLOCK_SIZE; lz++) {
                    for (int ly = 0; ly < BLOCK_SIZE; ly++) {
                        for (int lx = 0; lx < BLOCK_SIZE; lx++) {
                            // Compute world coordinate for OSM lookup
                            int vx = bk.x * BLOCK_SIZE + lx;
                            int vy = bk.y * BLOCK_SIZE + ly;
                            // int vz = bk.z * BLOCK_SIZE + lz; // z not needed for 2D OSM prior
                            float wx = (vx + 0.5f) * resolution_;
                            float wy = (vy + 0.5f) * resolution_;

                            // 1. Get OSM prior m_i (K_prior)
                            // Reuse buf_m_i
                            if (buf_m_i.size() != static_cast<size_t>(K_prior_))
                                buf_m_i.resize(static_cast<size_t>(K_prior_));
                            getOSMPrior(wx, wy, buf_m_i);

                            // 2. Compute OSM confidence c_xi = max(m_i)
                            float c_xi = 0.0f;
                            for (float v : buf_m_i) if (v > c_xi) c_xi = v;

                            // 3. Compute lambda(x)
                            float lambda = lambda_max_ - (lambda_max_ - lambda_min_) * c_xi;
                            
                            // 4. Compute effective lambda for dt steps
                            // If dt is large, pow might be slow. But dt is usually 1.
                            float lambda_eff = (dt == 1) ? lambda : std::pow(lambda, static_cast<float>(dt));

                            // 5. Compute alpha_osm(x)
                            // alpha_osm = alpha0 + gamma * p_pred
                            // We need p_pred. Reuse computePredPriorFromOSM logic but we already have m_i.
                            // Let's just call computePredPriorFromOSM with the buffers, it re-calls getOSMPrior.
                            // Optimization: computePredPriorFromOSM calls getOSMPrior internally. 
                            // We can just call it directly.
                            computePredPriorFromOSM(wx, wy, buf_p_pred, buf_m_i, buf_p_super);

                            // 6. Apply decay: alpha = lambda_eff * alpha + (1-lambda_eff) * alpha_osm
                            int idx_base = flatIndex(lx, ly, lz, 0);
                            for (int c = 0; c < K; c++) {
                                float alpha_osm_c = alpha0_;
                                if (seed_osm_prior_ && osm_prior_strength_ > 0.0f && buf_p_pred.size() == static_cast<size_t>(K)) {
                                    alpha_osm_c += osm_prior_strength_ * buf_p_pred[c];
                                }
                                
                                float& alpha_curr = blk.alpha[static_cast<size_t>(idx_base + c)];
                                alpha_curr = lambda_eff * alpha_curr + (1.0f - lambda_eff) * alpha_osm_c;
                            }
                        }
                    }
                }
            }
        }
        return it->second;
    }

    Block blk;
    const size_t total = static_cast<size_t>(BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * config_.num_total_classes);
    blk.alpha.resize(total, alpha0_);
    blk.last_updated = current_time; // Initialize with current time

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

const Block* ContinuousBKI::getBlockConst(const std::unordered_map<BlockKey, Block, BlockKeyHasher>& shard_map, const BlockKey& bk) const {
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

// computePredPriorFromOSM with pre-allocated buffers (no heap allocs)
void ContinuousBKI::computePredPriorFromOSM(float x, float y,
                                              std::vector<float>& p_pred_out,
                                              std::vector<float>& buf_m_i,
                                              std::vector<float>& buf_p_super) const {
    const int K = config_.num_total_classes;

    // Reuse buf_m_i
    if (buf_m_i.size() != static_cast<size_t>(K_prior_))
        buf_m_i.resize(static_cast<size_t>(K_prior_));
    getOSMPrior(x, y, buf_m_i);

    // Reuse buf_p_super
    if (buf_p_super.size() != static_cast<size_t>(K_pred_))
        buf_p_super.resize(static_cast<size_t>(K_pred_));
    std::fill(buf_p_super.begin(), buf_p_super.end(), 0.0f);

    for (int i = 0; i < K_pred_; i++) {
        float acc = 0.0f;
        for (int j = 0; j < K_prior_; j++) {
            acc += config_.confusion_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] * buf_m_i[j];
        }
        buf_p_super[i] = acc;
    }

    // Expand to full class space
    if (p_pred_out.size() != static_cast<size_t>(K))
        p_pred_out.resize(static_cast<size_t>(K));
    std::fill(p_pred_out.begin(), p_pred_out.end(), 0.0f);

    for (int i = 0; i < K_pred_; i++) {
        const auto& dense_labels = matrix_idx_to_dense_[static_cast<size_t>(i)];
        if (!dense_labels.empty()) {
            float share = buf_p_super[i] / static_cast<float>(dense_labels.size());
            for (int d : dense_labels) {
                if (d >= 0 && d < K) {
                    p_pred_out[static_cast<size_t>(d)] += share;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int c = 0; c < K; c++) sum += p_pred_out[c];
    if (sum > epsilon_)
        for (int c = 0; c < K; c++) p_pred_out[c] /= sum;
}

float ContinuousBKI::computeSpatialKernel(float dist_sq) const {
    if (!use_spatial_kernel_) return 1.0f;

    float dist = std::sqrt(dist_sq);
    float xi = dist / l_scale_;

    if (xi < 1.0f) {
        float term1 = (1.0f/3.0f) * (2.0f + std::cos(2.0f * static_cast<float>(M_PI) * xi)) * (1.0f - xi);
        float term2 = (1.0f/(2.0f * static_cast<float>(M_PI))) * std::sin(2.0f * static_cast<float>(M_PI) * xi);
        return sigma_0_ * (term1 + term2);
    }
    return 0.0f;
}

float ContinuousBKI::computeDistanceToClass(float x, float y, int class_idx) const {
    return computeDistanceToClassBruteForce(osm_data_, x, y, class_idx);
}

// getOSMPrior: uses precomputed raster when available, falls back to brute force
void ContinuousBKI::getOSMPrior(float x, float y, std::vector<float>& m_i) const {
    if (m_i.size() != static_cast<size_t>(K_prior_)) m_i.resize(static_cast<size_t>(K_prior_));

    if (osm_prior_raster_.width > 0) {
        osm_prior_raster_.lookup(x, y, m_i);
        return;
    }

    // Fallback: compute on the fly (no raster built)
    float sum = 0.0f;
    for (int k = 0; k < K_prior_; k++) {
        float dist = computeDistanceToClass(x, y, k);
        float score = 1.0f / (1.0f + std::exp((dist / delta_) - 4.6f));
        m_i[static_cast<size_t>(k)] = score;
        sum += score;
    }
    if (sum > epsilon_) {
        for (int k = 0; k < K_prior_; k++) m_i[k] /= sum;
    }
}

// getSemanticKernel with pre-allocated buffer for expected_obs
float ContinuousBKI::getSemanticKernel(int matrix_idx, const std::vector<float>& m_i,
                                        std::vector<float>& buf_expected_obs) const {
    if (!use_semantic_kernel_) return 1.0f;
    if (matrix_idx < 0 || matrix_idx >= K_pred_) return 1.0f;

    float c_xi = *std::max_element(m_i.begin(), m_i.end());

    if (buf_expected_obs.size() != static_cast<size_t>(K_pred_))
        buf_expected_obs.resize(static_cast<size_t>(K_pred_));
    std::fill(buf_expected_obs.begin(), buf_expected_obs.end(), 0.0f);

    for (int i = 0; i < K_pred_; i++) {
        float acc = 0.0f;
        for (int j = 0; j < K_prior_; j++) {
            acc += config_.confusion_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] * m_i[static_cast<size_t>(j)];
        }
        buf_expected_obs[i] = acc;
    }
    float numerator = buf_expected_obs[static_cast<size_t>(matrix_idx)];
    float denominator = *std::max_element(buf_expected_obs.begin(), buf_expected_obs.end()) + epsilon_;
    float s_i = numerator / denominator;
    return (1.0f - c_xi) + (c_xi * s_i);
}

// =====================================================================
// update (labels overload)
// =====================================================================
void ContinuousBKI::update(const std::vector<uint32_t>& labels, const std::vector<Point3D>& points) {
    if (labels.size() != points.size()) {
        std::cerr << "Mismatch in points and labels size" << std::endl;
        return;
    }

    size_t n = points.size();
    std::vector<float> point_k_sem(n, 1.0f);
    
    // Increment global time for lazy decay
    current_time_++;

    if (use_semantic_kernel_) {
#ifdef _OPENMP
#pragma omp parallel
        {
            // Thread-local buffers: no heap allocs inside the loop
            std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
            std::vector<float> tl_expected_obs(static_cast<size_t>(K_pred_));
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; i++) {
                getOSMPrior(points[i].x, points[i].y, tl_m_i);
                int raw_label = static_cast<int>(labels[i]);
                int matrix_idx = (raw_label >= 0 && raw_label <= max_raw_label_)
                                 ? label_to_matrix_flat_[static_cast<size_t>(raw_label)] : -1;
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
            int raw_label = static_cast<int>(labels[i]);
            int matrix_idx = (raw_label >= 0 && raw_label <= max_raw_label_)
                             ? label_to_matrix_flat_[static_cast<size_t>(raw_label)] : -1;
            if (matrix_idx >= 0) {
                point_k_sem[i] = getSemanticKernel(matrix_idx, tl_m_i, tl_expected_obs);
            }
        }
#endif
    }

    int num_shards = static_cast<int>(block_shards_.size());
    int radius = static_cast<int>(std::ceil(l_scale_ / resolution_));
    float l_scale_sq = l_scale_ * l_scale_;

    // Pre-compute which shards each point's influence region touches.
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

#ifdef _OPENMP
#pragma omp parallel num_threads(num_shards)
    {
        // Thread-local buffers for getOrCreateBlock -> initVoxelAlpha -> computePredPriorFromOSM
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

            int raw_label = static_cast<int>(labels[i]);
            int dense_label = (raw_label >= 0 && raw_label <= max_raw_label_)
                              ? raw_to_dense_flat_[static_cast<size_t>(raw_label)] : -1;
            if (dense_label < 0) continue;
            float k_sem = point_k_sem[i];

            for (int dx = -radius; dx <= radius; dx++) {
                int dy_limit = static_cast<int>(std::sqrt(std::max(0.0f,
                    static_cast<float>(radius * radius - dx * dx))));
                for (int dy = -dy_limit; dy <= dy_limit; dy++) {
                    int dz_limit = static_cast<int>(std::sqrt(std::max(0.0f,
                        static_cast<float>(radius * radius - dx * dx - dy * dy))));
                    for (int dz = -dz_limit; dz <= dz_limit; dz++) {
                        VoxelKey vk = {vk_p.x + dx, vk_p.y + dy, vk_p.z + dz};
                        BlockKey bk = voxelToBlockKey(vk);
                        if (getShardIndex(bk) != s) continue;

                        Point3D v_center = keyToPoint(vk);
                        float dist_sq = p.dist_sq(v_center);
                        if (dist_sq > l_scale_sq) continue;

                        float k_sp = computeSpatialKernel(dist_sq);
                        if (k_sp <= 1e-6f) continue;

                        Block& blk = getOrCreateBlock(shard, bk, tl_m_i, tl_p_super, tl_p_pred, current_time_);
                        int lx, ly, lz_local;
                        voxelToLocal(vk, lx, ly, lz_local);
                        int idx = flatIndex(lx, ly, lz_local, dense_label);
                        blk.alpha[static_cast<size_t>(idx)] += k_sp * k_sem;
                    }
                }
            }
        }
    }
#ifdef _OPENMP
    }
#endif
}

// =====================================================================
// update (probs overload)
// =====================================================================
void ContinuousBKI::update(const std::vector<std::vector<float>>& probs, const std::vector<Point3D>& points, const std::vector<float>& weights) {
    if (probs.size() != points.size()) {
        std::cerr << "Mismatch in points and probs size" << std::endl;
        return;
    }
    bool use_weights = !weights.empty();
    if (use_weights && weights.size() != points.size()) {
        std::cerr << "Mismatch in points and weights size" << std::endl;
        return;
    }

    size_t n = points.size();
    int num_shards = static_cast<int>(block_shards_.size());
    int radius = static_cast<int>(std::ceil(l_scale_ / resolution_));
    float l_scale_sq = l_scale_ * l_scale_;

    // Increment global time for lazy decay
    current_time_++;

    // Pre-compute which shards each point's influence region touches.
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

            const std::vector<float>& prob = probs[i];
            float w_i = use_weights ? weights[i] : 1.0f;

            for (int dx = -radius; dx <= radius; dx++) {
                int dy_limit = static_cast<int>(std::sqrt(std::max(0.0f,
                    static_cast<float>(radius * radius - dx * dx))));
                for (int dy = -dy_limit; dy <= dy_limit; dy++) {
                    int dz_limit = static_cast<int>(std::sqrt(std::max(0.0f,
                        static_cast<float>(radius * radius - dx * dx - dy * dy))));
                    for (int dz = -dz_limit; dz <= dz_limit; dz++) {
                        VoxelKey vk = {vk_p.x + dx, vk_p.y + dy, vk_p.z + dz};
                        BlockKey bk = voxelToBlockKey(vk);
                        if (getShardIndex(bk) != s) continue;

                        Point3D v_center = keyToPoint(vk);
                        float dist_sq = p.dist_sq(v_center);
                        if (dist_sq > l_scale_sq) continue;

                        float k_sp = computeSpatialKernel(dist_sq);
                        if (k_sp <= 1e-6f) continue;

                        Block& blk = getOrCreateBlock(shard, bk, tl_m_i, tl_p_super, tl_p_pred, current_time_);
                        int lx, ly, lz_local;
                        voxelToLocal(vk, lx, ly, lz_local);

                        for (size_t c = 0; c < prob.size(); c++) {
                            int idx = flatIndex(lx, ly, lz_local, static_cast<int>(c));
                            blk.alpha[static_cast<size_t>(idx)] += w_i * k_sp * prob[c];
                        }
                    }
                }
            }
        }
    }
#ifdef _OPENMP
    }
#endif
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
    const size_t n = points.size();
    std::vector<uint32_t> results(n, 0);

#ifdef _OPENMP
#pragma omp parallel
    {
        // Thread-local buffers for OSM fallback path
        std::vector<float> tl_p_pred(static_cast<size_t>(config_.num_total_classes));
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            const Point3D& p = points[i];
            VoxelKey k = pointToKey(p);
            BlockKey bk = voxelToBlockKey(k);
            int s = getShardIndex(bk);
            const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

            if (blk != nullptr) {
                int lx, ly, lz;
                voxelToLocal(k, lx, ly, lz);
                float sum = 0.0f;
                int best_idx = 0;
                float best_val = -1.0f;
                for (int c = 0; c < config_.num_total_classes; c++) {
                    float v = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                    sum += v;
                    if (v > best_val) { best_val = v; best_idx = c; }
                }
                if (sum > epsilon_) {
                    int raw = (best_idx >= 0 && best_idx < static_cast<int>(dense_to_raw_flat_.size()))
                              ? dense_to_raw_flat_[static_cast<size_t>(best_idx)] : -1;
                    results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                    continue;
                }
            }

            if (osm_fallback_in_infer_ && K_pred_ > 0) {
                computePredPriorFromOSM(p.x, p.y, tl_p_pred, tl_m_i, tl_p_super);
                if (!tl_p_pred.empty()) {
                    int best = static_cast<int>(std::max_element(tl_p_pred.begin(), tl_p_pred.end()) - tl_p_pred.begin());
                    if (best >= 0 && best < static_cast<int>(dense_to_raw_flat_.size())) {
                        int raw = dense_to_raw_flat_[static_cast<size_t>(best)];
                        results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                    }
                }
            }
        }
    }
#else
    std::vector<float> tl_p_pred(static_cast<size_t>(config_.num_total_classes));
    std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
    std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));

    for (size_t i = 0; i < n; i++) {
        const Point3D& p = points[i];
        VoxelKey k = pointToKey(p);
        BlockKey bk = voxelToBlockKey(k);
        int s = getShardIndex(bk);
        const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

        if (blk != nullptr) {
            int lx, ly, lz;
            voxelToLocal(k, lx, ly, lz);
            float sum = 0.0f;
            int best_idx = 0;
            float best_val = -1.0f;
            for (int c = 0; c < config_.num_total_classes; c++) {
                float v = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                sum += v;
                if (v > best_val) { best_val = v; best_idx = c; }
            }
            if (sum > epsilon_) {
                int raw = (best_idx >= 0 && best_idx < static_cast<int>(dense_to_raw_flat_.size()))
                          ? dense_to_raw_flat_[static_cast<size_t>(best_idx)] : -1;
                results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                continue;
            }
        }

        if (osm_fallback_in_infer_ && K_pred_ > 0) {
            computePredPriorFromOSM(p.x, p.y, tl_p_pred, tl_m_i, tl_p_super);
            if (!tl_p_pred.empty()) {
                int best = static_cast<int>(std::max_element(tl_p_pred.begin(), tl_p_pred.end()) - tl_p_pred.begin());
                if (best >= 0 && best < static_cast<int>(dense_to_raw_flat_.size())) {
                    int raw = dense_to_raw_flat_[static_cast<size_t>(best)];
                    results[i] = (raw >= 0) ? static_cast<uint32_t>(raw) : 0;
                }
            }
        }
    }
#endif

    return results;
}

// =====================================================================
// infer_probs - parallelized with OpenMP
// =====================================================================
std::vector<std::vector<float>> ContinuousBKI::infer_probs(const std::vector<Point3D>& points) const {
    const size_t n = points.size();
    const int K = config_.num_total_classes;
    std::vector<std::vector<float>> results(n);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<float> tl_probs(static_cast<size_t>(K));
        std::vector<float> tl_p_pred(static_cast<size_t>(K));
        std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
        std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
        std::vector<float> uniform(static_cast<size_t>(K), 1.0f / static_cast<float>(K));

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            const Point3D& p = points[i];
            VoxelKey k = pointToKey(p);
            BlockKey bk = voxelToBlockKey(k);
            int s = getShardIndex(bk);
            const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

            if (blk != nullptr) {
                int lx, ly, lz;
                voxelToLocal(k, lx, ly, lz);
                float sum = 0.0f;
                for (int c = 0; c < K; c++) {
                    tl_probs[c] = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                    sum += tl_probs[c];
                }
                if (sum > epsilon_) {
                    for (int c = 0; c < K; c++) tl_probs[c] /= sum;
                    results[i] = tl_probs;
                    continue;
                }
            }

            if (osm_fallback_in_infer_ && K_pred_ > 0) {
                computePredPriorFromOSM(p.x, p.y, tl_p_pred, tl_m_i, tl_p_super);
                if (tl_p_pred.size() == static_cast<size_t>(K)) {
                    results[i] = tl_p_pred;
                } else {
                    results[i] = uniform;
                }
            } else {
                results[i] = uniform;
            }
        }
    }
#else
    std::vector<float> tl_probs(static_cast<size_t>(K));
    std::vector<float> tl_p_pred(static_cast<size_t>(K));
    std::vector<float> tl_m_i(static_cast<size_t>(K_prior_));
    std::vector<float> tl_p_super(static_cast<size_t>(K_pred_));
    std::vector<float> uniform(static_cast<size_t>(K), 1.0f / static_cast<float>(K));

    for (size_t i = 0; i < n; i++) {
        const Point3D& p = points[i];
        VoxelKey k = pointToKey(p);
        BlockKey bk = voxelToBlockKey(k);
        int s = getShardIndex(bk);
        const Block* blk = getBlockConst(block_shards_[static_cast<size_t>(s)], bk);

        if (blk != nullptr) {
            int lx, ly, lz;
            voxelToLocal(k, lx, ly, lz);
            float sum = 0.0f;
            for (int c = 0; c < K; c++) {
                tl_probs[c] = blk->alpha[static_cast<size_t>(flatIndex(lx, ly, lz, c))];
                sum += tl_probs[c];
            }
            if (sum > epsilon_) {
                for (int c = 0; c < K; c++) tl_probs[c] /= sum;
                results[i] = tl_probs;
                continue;
            }
        }

        if (osm_fallback_in_infer_ && K_pred_ > 0) {
            computePredPriorFromOSM(p.x, p.y, tl_p_pred, tl_m_i, tl_p_super);
            if (tl_p_pred.size() == static_cast<size_t>(K)) {
                results[i] = tl_p_pred;
            } else {
                results[i] = uniform;
            }
        } else {
            results[i] = uniform;
        }
    }
#endif

    return results;
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

    {
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

    std::cout << "OSM XML: Flat-earth projection, origin ("
              << origin_lat << ", " << origin_lon
              << "), world offset (" << offset_x << ", " << offset_y << ")" << std::endl;

    std::map<std::string, Point2D> node_coords;

    for (const auto& kv : parser.nodes) {
        auto xy = osm_xml_parser::latlon_to_meters(
            kv.second.lat, kv.second.lon, origin_lat, origin_lon);
        double x = xy.first + offset_x;
        double y = xy.second + offset_y;
        node_coords[kv.first] = Point2D(static_cast<float>(x), static_cast<float>(y));
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
        config.label_to_matrix_idx = yaml.getLabelToMatrixIdx();
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

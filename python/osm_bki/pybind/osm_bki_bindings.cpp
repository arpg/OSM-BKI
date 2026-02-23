/**
 * Python bindings for OSM-S-BKI (composite_bki_cpp).
 * Exposes PyContinuousBKI matching the Cython bindings.pyx API
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "continuous_bki.hpp"
#include "osm_loader.hpp"
#include "osm_xml_parser.hpp"
#include "pose_utils.hpp"

namespace py = pybind11;

namespace {

using namespace continuous_bki;

void compute_metrics(const uint32_t* refined, const uint32_t* gt, size_t n,
                     double& accuracy, double& miou) {
    size_t correct = 0;
    for (size_t i = 0; i < n; ++i)
        if (refined[i] == gt[i]) ++correct;
    accuracy = n > 0 ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;

    std::unordered_map<uint32_t, int64_t> tp, fp, fn;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = refined[i], g = gt[i];
        if (r == g) { tp[r] += 1; } else { fp[r] += 1; fn[g] += 1; }
    }
    std::unordered_map<uint32_t, int64_t> all_classes;
    for (const auto& kv : tp) all_classes[kv.first];
    for (const auto& kv : fp) all_classes[kv.first];
    for (const auto& kv : fn) all_classes[kv.first];

    double sum_iou = 0.0;
    int num_classes = 0;
    for (const auto& kv : all_classes) {
        uint32_t c = kv.first;
        int64_t tp_c = tp.count(c) ? tp[c] : 0;
        int64_t fp_c = fp.count(c) ? fp[c] : 0;
        int64_t fn_c = fn.count(c) ? fn[c] : 0;
        int64_t denom = tp_c + fp_c + fn_c;
        if (denom > 0) {
            sum_iou += static_cast<double>(tp_c) / static_cast<double>(denom);
            ++num_classes;
        }
    }
    miou = num_classes > 0 ? sum_iou / static_cast<double>(num_classes) : 0.0;
}

std::vector<Point3D> array_to_points(const py::buffer_info& pb) {
    size_t n = static_cast<size_t>(pb.shape[0]);
    const float* px = static_cast<const float*>(pb.ptr);
    // Use strides to support C-order, F-order, and arbitrary views (e.g. scan[:, :3] or transform output)
    py::ssize_t s0 = pb.strides[0] / static_cast<py::ssize_t>(sizeof(float));
    py::ssize_t s1 = pb.strides[1] / static_cast<py::ssize_t>(sizeof(float));
    std::vector<Point3D> pts(n);
    for (size_t i = 0; i < n; ++i)
        pts[i] = Point3D(px[i * s0 + 0 * s1], px[i * s0 + 1 * s1], px[i * s0 + 2 * s1]);
    return pts;
}

}  // namespace


// --- PyContinuousBKI: matches the PyContinuousBKI class in bindings.pyx
class PyContinuousBKI {
public:
    PyContinuousBKI(const std::string& osm_path,
                    const std::string& config_path,
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
                    bool osm_fallback_in_infer = true)
        : config_(loadConfigFromYAML(config_path)),
          osm_data_(loadOSM(osm_path, config_)),
          bki_(config_, osm_data_,
               resolution, l_scale, sigma_0, prior_delta, height_sigma,
               use_semantic_kernel, use_spatial_kernel, num_threads,
               alpha0, seed_osm_prior, osm_prior_strength, osm_fallback_in_infer) {}

    // Hard labels update: update(labels, points) — labels first, matching Cython arg order
    void update(py::array_t<uint32_t> labels, py::array_t<float> points) {
        py::buffer_info pb = points.request();
        py::buffer_info lb = labels.request();
        if (pb.ndim != 2 || pb.shape[1] < 3)
            throw std::runtime_error("points must be (N, 3+) float32");
        if (lb.ndim != 1)
            throw std::runtime_error("labels must be (N,) uint32");
        size_t n = static_cast<size_t>(pb.shape[0]);
        if (lb.shape[0] != static_cast<py::ssize_t>(n))
            throw std::runtime_error("labels and points length mismatch");

        std::vector<uint32_t> lbls(n);
        const char* lbl_base = static_cast<const char*>(lb.ptr);
        for (size_t i = 0; i < n; ++i)
            lbls[i] = *reinterpret_cast<const uint32_t*>(lbl_base + i * lb.strides[0]);
        bki_.update(lbls, array_to_points(pb));
    }

    // Soft/probabilistic update: update_soft(probs, points, weights=None)
    void update_soft(py::array_t<float> probs, py::array_t<float> points,
                     py::object weights = py::none()) {
        py::buffer_info pb = points.request();
        py::buffer_info rb = probs.request();
        if (pb.ndim != 2 || pb.shape[1] < 3)
            throw std::runtime_error("points must be (N, 3+) float32");
        if (rb.ndim != 2)
            throw std::runtime_error("probs must be (N, num_classes) float32");
        size_t n = static_cast<size_t>(pb.shape[0]);
        if (rb.shape[0] != static_cast<py::ssize_t>(n))
            throw std::runtime_error("probs and points length mismatch");

        const float* rx = static_cast<const float*>(rb.ptr);
        int num_classes = static_cast<int>(rb.shape[1]);
        py::ssize_t r0 = rb.strides[0] / static_cast<py::ssize_t>(sizeof(float));
        py::ssize_t r1 = rb.strides[1] / static_cast<py::ssize_t>(sizeof(float));
        std::vector<std::vector<float>> cpp_probs(n);
        for (size_t i = 0; i < n; ++i) {
            cpp_probs[i].reserve(num_classes);
            for (int j = 0; j < num_classes; ++j)
                cpp_probs[i].push_back(rx[i * r0 + j * r1]);
        }

        std::vector<float> cpp_weights;
        if (!weights.is_none()) {
            py::array_t<float> w = py::cast<py::array_t<float>>(weights);
            py::buffer_info wb = w.request();
            if (wb.ndim != 1 || wb.shape[0] != static_cast<py::ssize_t>(n))
                throw std::runtime_error("weights must be (N,) float32");
            cpp_weights.resize(n);
            for (size_t i = 0; i < n; ++i)
                cpp_weights[i] = *reinterpret_cast<const float*>(
                    static_cast<const char*>(wb.ptr) + i * wb.strides[0]);
        }

        bki_.update(cpp_probs, array_to_points(pb), cpp_weights);
    }

    // Hard label inference: infer(points) -> (N,) uint32
    py::array_t<uint32_t> infer(py::array_t<float> points) {
        py::buffer_info pb = points.request();
        if (pb.ndim != 2 || pb.shape[1] < 3)
            throw std::runtime_error("points must be (N, 3+) float32");
        size_t n = static_cast<size_t>(pb.shape[0]);
        std::vector<uint32_t> result = bki_.infer(array_to_points(pb));

        py::array_t<uint32_t> out(static_cast<py::ssize_t>(n));
        std::copy(result.begin(), result.end(), static_cast<uint32_t*>(out.request().ptr));
        return out;
    }

    // Probability inference: infer_probs(points) -> (N, num_classes) float32
    py::array_t<float> infer_probs(py::array_t<float> points) {
        py::buffer_info pb = points.request();
        if (pb.ndim != 2 || pb.shape[1] < 3)
            throw std::runtime_error("points must be (N, 3+) float32");
        size_t n = static_cast<size_t>(pb.shape[0]);
        std::vector<std::vector<float>> result = bki_.infer_probs(array_to_points(pb));

        if (result.empty() || result[0].empty())
            return py::array_t<float>(std::vector<py::ssize_t>{0, 0});

        size_t num_classes = result[0].size();
        py::array_t<float> out({(py::ssize_t)n, (py::ssize_t)num_classes});
        float* out_ptr = static_cast<float*>(out.request().ptr);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < num_classes; ++j)
                out_ptr[i * num_classes + j] = result[i][j];
        return out;
    }

    int get_size() { return bki_.size(); }
    void clear() { bki_.clear(); }
    void save(const std::string& filename) { bki_.save(filename); }
    void load(const std::string& filename) { bki_.load(filename); }
    void print_profiling_stats() { bki_.printProfilingStats(); }

    // evaluate_metrics: convenience method not in Cython but non-conflicting
    py::dict evaluate_metrics(py::array_t<uint32_t> refined, py::array_t<uint32_t> gt) {
        py::buffer_info rb = refined.request();
        py::buffer_info gb = gt.request();
        if (rb.ndim != 1 || gb.ndim != 1 || rb.shape[0] != gb.shape[0])
            throw std::runtime_error("refined and gt must be 1D arrays of same length");
        double accuracy, miou;
        compute_metrics(static_cast<const uint32_t*>(rb.ptr),
                        static_cast<const uint32_t*>(gb.ptr),
                        static_cast<size_t>(rb.shape[0]), accuracy, miou);
        py::dict d;
        d["accuracy"] = accuracy;
        d["miou"] = miou;
        return d;
    }

private:
    Config config_;
    OSMData osm_data_;
    ContinuousBKI bki_;
};

PYBIND11_MODULE(composite_bki_cpp, m) {
    m.doc() = "OSM-S-BKI / Composite BKI C++ extension: semantic BKI with OSM priors";

    m.def("load_osm_geometries",
          [](const std::string& osm_path, const std::string& config_path, double z_offset) {
              Config config = loadConfigFromYAML(config_path);
              OSMData data = loadOSM(osm_path, config);

              std::map<int, std::string> idx_to_name;
              for (const auto& kv : config.osm_class_map) {
                  idx_to_name[kv.second] = kv.first;
              }

              auto get_color = [](const std::string& cat) -> std::vector<double> {
                  if (cat == "buildings") return {30/255., 180/255., 30/255.};
                  if (cat == "roads") return {240/255., 120/255., 20/255.};
                  if (cat == "sidewalks") return {220/255., 220/255., 220/255.};
                  if (cat == "parking") return {245/255., 210/255., 80/255.};
                  if (cat == "fences" || cat == "barriers" || cat == "poles" || cat == "traffic_signs")
                      return {170/255., 120/255., 70/255.};
                  if (cat == "grasslands" || cat == "trees" || cat == "wood")
                      return {60/255., 170/255., 80/255.};
                  return {140/255., 140/255., 140/255.};
              };

              py::list result;
              for (const auto& kv : data.geometries) {
                  int class_idx = kv.first;
                  std::string cat = idx_to_name.count(class_idx) ? idx_to_name[class_idx] : "default";
                  std::vector<double> color = get_color(cat);

                  std::vector<std::vector<double>> all_points;
                  std::vector<std::vector<int>> all_lines;
                  size_t point_offset = 0;

                  for (const auto& poly : kv.second) {
                      if (poly.points.size() < 2) continue;
                      for (const auto& p : poly.points) {
                          all_points.push_back({static_cast<double>(p.x), static_cast<double>(p.y), z_offset});
                      }
                      size_t n = poly.points.size();
                      for (size_t i = 0; i < n; i++) {
                          size_t j = (i + 1) % n;
                          all_lines.push_back({static_cast<int>(point_offset + i), static_cast<int>(point_offset + j)});
                      }
                      point_offset += n;
                  }

                  if (all_points.empty()) continue;

                  py::dict d;
                  d["points"] = all_points;
                  d["lines"] = all_lines;
                  d["color"] = color;
                  result.append(d);
              }
              return result;
          },
          py::arg("osm_path"),
          py::arg("config_path"),
          py::arg("z_offset") = 0.05,
          "Load OSM file and return list of {points, lines, color} dicts for Open3D visualization.");

    m.def("latlon_to_mercator",
          [](double lat, double lon, double origin_lat, double origin_lon,
             double world_offset_x, double world_offset_y) {
              auto xy = osm_xml_parser::latlon_to_mercator(
                  lat, lon, origin_lat, origin_lon, world_offset_x, world_offset_y);
              return py::make_tuple(xy.first, xy.second);
          },
          py::arg("lat"), py::arg("lon"),
          py::arg("origin_lat"), py::arg("origin_lon"),
          py::arg("world_offset_x") = 0.0, py::arg("world_offset_y") = 0.0,
          "Convert lat/lon to local metres using scaled Mercator projection.");

    m.def("transform_scan_to_world",
          [](py::array_t<float> points,
             py::array_t<double> pose,
             py::array_t<double> body_to_lidar,
             py::object init_rel_pos_obj) -> py::array_t<float> {
              py::buffer_info pb = points.request();
              if (pb.ndim != 2 || pb.shape[1] < 3)
                  throw std::runtime_error("points must be (N, 3+) float32");

              py::buffer_info poseb = pose.request();
              if (poseb.ndim != 1 || poseb.shape[0] < 7)
                  throw std::runtime_error("pose must be (7,) float64: [x, y, z, qx, qy, qz, qw]");

              py::buffer_info cb = body_to_lidar.request();
              if (cb.ndim != 2 || cb.shape[0] != 4 || cb.shape[1] != 4)
                  throw std::runtime_error("body_to_lidar must be (4, 4) float64");

              const double* pd = static_cast<const double*>(poseb.ptr);

              continuous_bki::Transform4x4 calib;
              const double* cd = static_cast<const double*>(cb.ptr);
              py::ssize_t cs0 = cb.strides[0] / static_cast<py::ssize_t>(sizeof(double));
              py::ssize_t cs1 = cb.strides[1] / static_cast<py::ssize_t>(sizeof(double));
              for (int i = 0; i < 4; ++i)
                  for (int j = 0; j < 4; ++j)
                      calib.m[i * 4 + j] = cd[i * cs0 + j * cs1];

              const double* irp = nullptr;
              double irp_buf[3];
              if (!init_rel_pos_obj.is_none()) {
                  py::array_t<double> irp_arr = py::cast<py::array_t<double>>(init_rel_pos_obj);
                  py::buffer_info ib = irp_arr.request();
                  if (ib.ndim != 1 || ib.shape[0] < 3)
                      throw std::runtime_error("init_rel_pos must be (3,) float64");
                  const double* id = static_cast<const double*>(ib.ptr);
                  irp_buf[0] = id[0]; irp_buf[1] = id[1]; irp_buf[2] = id[2];
                  irp = irp_buf;
              }

              std::vector<continuous_bki::Point3D> pts = array_to_points(pb);
              std::vector<continuous_bki::Point3D> out =
                  continuous_bki::transformScanToWorld(
                      pts, pd[0], pd[1], pd[2],
                      pd[3], pd[4], pd[5], pd[6],
                      calib, irp);

              size_t n = out.size();
              py::array_t<float> result({(py::ssize_t)n, (py::ssize_t)3});
              float* rp = static_cast<float*>(result.request().ptr);
              for (size_t i = 0; i < n; ++i) {
                  rp[i * 3 + 0] = out[i].x;
                  rp[i * 3 + 1] = out[i].y;
                  rp[i * 3 + 2] = out[i].z;
              }
              return result;
          },
          py::arg("points"),
          py::arg("pose"),
          py::arg("body_to_lidar"),
          py::arg("init_rel_pos") = py::none(),
          "Transform LiDAR points (N,3) float32 to world frame.\n"
          "pose: (7,) float64 [x,y,z,qx,qy,qz,qw]\n"
          "body_to_lidar: (4,4) float64 calibration matrix\n"
          "init_rel_pos: (3,) float64 or None -- subtracted from pose translation");

    py::class_<PyContinuousBKI>(m, "PyContinuousBKI")
        .def(py::init<const std::string&, const std::string&,
                      float, float, float, float, float,
                      bool, bool, int,
                      float, bool, float, bool>(),
             py::arg("osm_path"),
             py::arg("config_path"),
             py::arg("resolution") = 0.1f,
             py::arg("l_scale") = 0.3f,
             py::arg("sigma_0") = 1.0f,
             py::arg("prior_delta") = 5.0f,
             py::arg("height_sigma") = 0.3f,
             py::arg("use_semantic_kernel") = true,
             py::arg("use_spatial_kernel") = true,
             py::arg("num_threads") = -1,
             py::arg("alpha0") = 1.0f,
             py::arg("seed_osm_prior") = false,
             py::arg("osm_prior_strength") = 0.0f,
             py::arg("osm_fallback_in_infer") = true)
        .def("update", &PyContinuousBKI::update,
             py::arg("labels"), py::arg("points"),
             "Update BKI with hard labels and points.")
        .def("update_soft", &PyContinuousBKI::update_soft,
             py::arg("probs"), py::arg("points"), py::arg("weights") = py::none(),
             "Update BKI with soft label probabilities, points, and optional weights.")
        .def("infer", &PyContinuousBKI::infer,
             py::arg("points"),
             "Infer hard labels for points; returns (N,) uint32 array.")
        .def("infer_probs", &PyContinuousBKI::infer_probs,
             py::arg("points"),
             "Infer class probabilities for points; returns (N, num_classes) float32 array.")
        .def("get_size", &PyContinuousBKI::get_size,
             "Return number of voxels in the map.")
        .def("clear", &PyContinuousBKI::clear,
             "Clear the BKI map.")
        .def("save", &PyContinuousBKI::save,
             py::arg("filename"),
             "Save the BKI map to a binary file.")
        .def("load", &PyContinuousBKI::load,
             py::arg("filename"),
             "Load a BKI map from a binary file.")
        .def("evaluate_metrics", &PyContinuousBKI::evaluate_metrics,
             py::arg("refined"), py::arg("gt"),
             "Return dict with 'accuracy' and 'miou'.")
        .def("print_profiling_stats", &PyContinuousBKI::print_profiling_stats,
             "Print cumulative profiling statistics for raster build, update, and inference phases.");
}

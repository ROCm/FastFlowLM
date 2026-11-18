// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

/// @file bindings.cpp
/// @brief Python extension module (smolvla_npu) exposing SmolVLA::forward.
///
/// The NPU state is the reason this file exists in the shape it does. The
/// object graph is
///
///     xrt::device -> npu_xclbin_manager -> SmolVLA -> {vision,language,expert} exec
///
/// and every arrow is a raw pointer the callee caches: npu_xclbin_manager keeps
/// an xrt::device*, and each exec context keeps the npu_app_manager* it got from
/// register_xclbin plus bo-backed buffers carved out of it. So the chain must be
/// constructed once, never copied or moved, and destroyed in exactly the reverse
/// order. XRT also permits a single hardware context per device, so a second
/// live model would fail at xclbin registration. PySmolVLA owns the whole chain
/// and refuses to build a second one while the first is alive.
///
/// Torch tensors are handled through the Python C API only (data_ptr(), numel(),
/// is_contiguous()) rather than by linking libtorch: it keeps the module free of
/// any torch ABI coupling, and it is necessary regardless, since torch.bfloat16
/// tensors expose neither .numpy() nor the buffer protocol -- the f32<->bf16
/// conversion has to run in our own loop either way.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "npu_utils/npu_utils.hpp"
#include "utils/utils.hpp"
#include "SmolVLA/SmolVLA.hpp"
#include "SmolVLA/SmolVLA_config.hpp"

namespace py = pybind11;

namespace {

// --------------------------------------------------------------------------
// torch tensor access helpers (no libtorch link -- pure Python C API)
// --------------------------------------------------------------------------

/// @brief number of elements in a torch tensor
size_t tensor_numel(const py::object& t) {
    return t.attr("numel")().cast<size_t>();
}

/// @brief true if the tensor's dtype matches torch.<name>
bool tensor_dtype_is(const py::object& t, const char* name) {
    py::object want = py::module_::import("torch").attr(name);
    py::object got  = t.attr("dtype");
    return got.equal(want);
}

/// @brief raw pointer to a tensor's storage, after checking dtype/contiguity.
///
/// A non-contiguous tensor has a data_ptr that does not describe a flat dense
/// array, so reading numel elements from it would silently produce garbage.
/// Reject it instead and tell the caller to call .contiguous().
const void* tensor_data(const py::object& t, const char* dtype, const std::string& what) {
    if (!py::hasattr(t, "data_ptr")) {
        throw std::invalid_argument(what + ": expected a torch.Tensor");
    }
    if (!tensor_dtype_is(t, dtype)) {
        std::string got = py::str(t.attr("dtype")).cast<std::string>();
        throw std::invalid_argument(what + ": expected dtype torch." + dtype + ", got " + got);
    }
    if (!t.attr("is_contiguous")().cast<bool>()) {
        throw std::invalid_argument(what + ": tensor must be contiguous (call .contiguous())");
    }
    if (t.attr("device").attr("type").cast<std::string>() != "cpu") {
        throw std::invalid_argument(what + ": tensor must live on the CPU (call .cpu())");
    }
    return reinterpret_cast<const void*>(t.attr("data_ptr")().cast<uintptr_t>());
}

/// @brief copy a float32 torch tensor into a freshly allocated bf16 buffer
buffer<bf16> f32_tensor_to_bf16(const py::object& t, size_t expect_numel,
                                const std::string& what) {
    const size_t n = tensor_numel(t);
    if (n != expect_numel) {
        throw std::invalid_argument(what + ": expected " + std::to_string(expect_numel)
                                    + " elements, got " + std::to_string(n));
    }
    const float* src = static_cast<const float*>(tensor_data(t, "float32", what));
    buffer<bf16> dst(n);
    for (size_t i = 0; i < n; i++) dst[i] = (bf16)src[i];
    return dst;
}

/// @brief read a token id sequence from an int32/int64 tensor or a Python list
std::vector<int> to_token_vector(const py::object& tokens) {
    if (py::hasattr(tokens, "data_ptr")) {
        const size_t n = tensor_numel(tokens);
        std::vector<int> out(n);
        if (tensor_dtype_is(tokens, "int64")) {
            const int64_t* p = static_cast<const int64_t*>(
                tensor_data(tokens, "int64", "tokens"));
            for (size_t i = 0; i < n; i++) out[i] = (int)p[i];
        } else if (tensor_dtype_is(tokens, "int32")) {
            const int32_t* p = static_cast<const int32_t*>(
                tensor_data(tokens, "int32", "tokens"));
            for (size_t i = 0; i < n; i++) out[i] = (int)p[i];
        } else {
            std::string got = py::str(tokens.attr("dtype")).cast<std::string>();
            throw std::invalid_argument("tokens: expected dtype torch.int32 or torch.int64, got "
                                        + got);
        }
        return out;
    }
    // fall back to any Python sequence of ints
    return tokens.cast<std::vector<int>>();
}

/// @brief allocate a torch.float32 tensor of the given shape and fill it from bf16
py::object bf16_to_f32_tensor(const buffer<bf16>& src, size_t rows, size_t cols) {
    py::module_ torch = py::module_::import("torch");
    py::object out = torch.attr("empty")(
        py::make_tuple(rows, cols), py::arg("dtype") = torch.attr("float32"));
    float* dst = reinterpret_cast<float*>(out.attr("data_ptr")().cast<uintptr_t>());
    const bf16* p = src.data();
    for (size_t i = 0; i < rows * cols; i++) dst[i] = (float)p[i];
    return out;
}

// --------------------------------------------------------------------------
// PySmolVLA
// --------------------------------------------------------------------------

/// @brief Python-owned SmolVLA policy: the xrt device, the xclbin manager and
///        the model, held together so their lifetimes cannot come apart.
class PySmolVLA {
public:
    /// @brief build the whole NPU chain and load the weights
    /// @param model_path model directory (config.json, model.safetensors)
    explicit PySmolVLA(const std::string& model_path) {
        if (g_instance != nullptr) {
            throw std::runtime_error(
                "a SmolVLA instance already exists; only one is allowed at a time "
                "(delete the existing one first)");
        }
        // Rebuilding after a forward does not work: the second instance
        // constructs and loads weights fine, but its first NPU run dies with
        // ERT_CMD_STATE_TIMEOUT. This is not a binding artifact -- a pure C++
        // program that constructs, forwards, tears down and reconstructs fails
        // identically, including when the xrt::device is shared across rounds,
        // so something in the xclbin/hw_context re-registration path does not
        // survive a completed run. Refuse here with an explanation rather than
        // let the caller hit an opaque timeout deep inside forward().
        if (g_npu_used) {
            throw std::runtime_error(
                "the NPU cannot be re-initialized after a forward() in the same "
                "process (the second hardware context times out); start a new "
                "process to load another model");
        }
        // The model code exits the process on a missing file rather than
        // throwing, which would take the interpreter down with it. Check the
        // two paths it needs up front so the common mistake raises instead.
        // (utils::check_file_exists opens an ifstream, which does not reliably
        // distinguish a directory, so use std::filesystem here.) The xclbins
        // are not among them: they ship inside this package, and the resolver
        // that finds them already throws with the candidates it tried.
        for (const char* required : {"config.json", "model.safetensors"}) {
            const std::string p = utils::path_join(model_path, required);
            if (!std::filesystem::exists(p)) {
                throw std::runtime_error("model path is missing " + std::string(required)
                                         + ": " + p);
            }
        }

        this->device = xrt::device(0);   // the NPU is always device 0
        this->mgr    = std::make_unique<npu_xclbin_manager>(
            npu_device::device_npu2, &this->device, false);
        this->model  = std::make_unique<SmolVLA>(this->mgr.get());

        this->model->from_pretrained(model_path);
        this->model->load_weights();

        this->model_path_ = model_path;
        this->chunk_size_ = (int)this->model->expert->def.chunk_size;
        this->act_dim_    = (int)this->model->action_in_proj.shape[0];
        this->state_dim_  = (int)this->model->state_proj.shape[0];

        g_instance = this;
    }

    ~PySmolVLA() {
        // Destruction runs in reverse declaration order below (model, then mgr,
        // then device), which is exactly the order the raw back-pointers need.
        if (g_instance == this) g_instance = nullptr;
    }

    PySmolVLA(const PySmolVLA&) = delete;
    PySmolVLA& operator=(const PySmolVLA&) = delete;

    /// @brief full policy step: vlm prefill + flow-matching denoise.
    /// @param images one float32 [3, 512, 512] (or [1, 3, 512, 512]) tensor per camera
    /// @param tokens tokenizer-padded prompt ids (int32/int64 tensor or list)
    /// @param state  float32 robot state, up to state_dim elements (zero-padded)
    /// @param noise  optional float32 [chunk_size, action_dim] seed; sampled if None
    /// @return the denoised action chunk, float32 [chunk_size, action_dim]
    py::object forward(const py::list& images, const py::object& tokens,
                       const py::object& state, const py::object& noise) {
        // The NPU state (bo buffers, kv cache, exec scratch) is shared across a
        // forward, so two concurrent calls would interleave onto the same
        // hardware. Serialize instead of corrupting.
        std::lock_guard<std::mutex> lock(this->mu);

        const int num_images = (int)images.size();
        if (num_images < 1 || num_images > smolvla_cfg::MAX_IMAGES) {
            throw std::invalid_argument("images: expected 1.." +
                std::to_string(smolvla_cfg::MAX_IMAGES) + " camera tensors, got " +
                std::to_string(num_images));
        }

        constexpr size_t pixels_per_image = (size_t)smolvla_cfg::PATCH_CIN
                                          * smolvla_cfg::IMAGE_SIZE
                                          * smolvla_cfg::IMAGE_SIZE;

        smolvla_input_t input;
        // buffer<T>'s copy ctor is shallow, so size the vector first and
        // move-assign each owning buffer into place (same pattern as
        // SmolVLA_vision::alloc_weights).
        input.pixel_values.resize(num_images);
        for (int i = 0; i < num_images; i++) {
            input.pixel_values[i] = f32_tensor_to_bf16(
                images[i].cast<py::object>(), pixels_per_image,
                "images[" + std::to_string(i) + "]");
        }

        // Token ids pass through unchanged: the language prefill stops at the
        // first PAD_TOKEN_ID itself, so the tokenizer's padded output is exactly
        // what it wants.
        input.input_tokens = to_token_vector(tokens);
        if (input.input_tokens.empty()) {
            throw std::invalid_argument("tokens: sequence is empty");
        }

        // SmolVLA::predict zero-pads the state up to state_dim, so a shorter
        // (real) observation vector is fine; a longer one is not.
        const size_t state_n = tensor_numel(state);
        if (state_n == 0 || state_n > (size_t)this->state_dim_) {
            throw std::invalid_argument("state: expected 1.." + std::to_string(this->state_dim_)
                                        + " elements, got " + std::to_string(state_n));
        }
        input.states = f32_tensor_to_bf16(state, state_n, "state");

        const size_t noise_n = (size_t)this->chunk_size_ * this->act_dim_;
        buffer<bf16> noise_buf;
        buffer<bf16>* noise_ptr = nullptr;
        if (!noise.is_none()) {
            noise_buf = f32_tensor_to_bf16(noise, noise_n, "noise");
            noise_ptr = &noise_buf;
        }

        buffer<bf16> x;
        {
            // A forward is hundreds of milliseconds of NPU + OpenMP work that
            // touches nothing Python; hold the GIL only for the conversions.
            py::gil_scoped_release release;
            g_npu_used = true;   // latches: the NPU can no longer be re-initialized
            x = this->model->forward(input, noise_ptr, /*skip_prefill=*/false);
        }

        return bf16_to_f32_tensor(x, (size_t)this->chunk_size_, (size_t)this->act_dim_);
    }

    const std::string& model_path() const { return this->model_path_; }
    int chunk_size() const { return this->chunk_size_; }
    int action_dim() const { return this->act_dim_; }
    int state_dim()  const { return this->state_dim_; }

private:
    static PySmolVLA* g_instance;
    /// set once the NPU has actually run; never cleared (see the constructor).
    static bool g_npu_used;

    std::mutex mu;

    // Declaration order is load-bearing: members are destroyed in reverse, so
    // the model goes first, then the manager, then the device it points at.
    xrt::device device;
    std::unique_ptr<npu_xclbin_manager> mgr;
    std::unique_ptr<SmolVLA> model;

    std::string model_path_;
    int chunk_size_ = 0;
    int act_dim_    = 0;
    int state_dim_  = 0;
};

PySmolVLA* PySmolVLA::g_instance = nullptr;
bool PySmolVLA::g_npu_used = false;

/// @brief warn once at import if the CPU kernels have no OpenMP thread team.
///
/// main.cpp blocks on stdin here; a library must not, so this only reports.
void report_openmp() {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads > 1) return;
    py::module_::import("warnings").attr("warn")(
        "smolvla_npu: OpenMP is enabled but only 1 thread is available "
        "(check OMP_NUM_THREADS / cpu affinity); the CPU kernels will be slow.");
#else
    py::module_::import("warnings").attr("warn")(
        "smolvla_npu: built without OpenMP; the CPU kernels will run single "
        "threaded and will be slow.");
#endif
}

}  // namespace

PYBIND11_MODULE(smolvla_npu, m) {
    m.doc() = "SmolVLA policy running on the AMD NPU.\n\n"
              "Only one SmolVLA instance may exist at a time: it owns the XRT device\n"
              "and the NPU hardware context, which cannot be shared.\n\n"
              "Once forward() has run, the NPU cannot be re-initialized in the same\n"
              "process -- deleting the instance and constructing a new one raises.\n"
              "Load one model per process.";

    py::class_<PySmolVLA>(m, "SmolVLA")
        .def(py::init<const std::string&>(), py::arg("model_path"),
             "Load a SmolVLA policy from a model directory (config.json,\n"
             "model.safetensors) and bring up the NPU. The NPU bitstreams ship\n"
             "with this package; the model directory does not supply them.\n\n"
             "Raises RuntimeError if another instance is already alive, or if a\n"
             "forward() has already run in this process (the NPU cannot be\n"
             "re-initialized -- use a fresh process).")
        .def("forward", &PySmolVLA::forward,
             py::arg("images"), py::arg("tokens"), py::arg("state"),
             py::arg("noise") = py::none(),
             "Run one full policy step: VLM prefill over the observation, then\n"
             "the 10-step flow-matching denoise over the action expert.\n\n"
             "Args:\n"
             "    images: list of float32 CPU tensors, one per camera, each\n"
             "        [3, 512, 512] or [1, 3, 512, 512], already resized/padded\n"
             "        and scaled to [-1, 1].\n"
             "    tokens: prompt token ids, int32/int64 tensor or list of int.\n"
             "    state:  float32 robot state, up to state_dim elements\n"
             "        (zero-padded internally).\n"
             "    noise:  optional float32 [chunk_size, action_dim] seed for the\n"
             "        denoise loop. Sampled from the global RNG (see seed()) if\n"
             "        omitted.\n\n"
             "Returns:\n"
             "    float32 tensor [chunk_size, action_dim].")
        .def_property_readonly("model_path", &PySmolVLA::model_path)
        .def_property_readonly("chunk_size", &PySmolVLA::chunk_size,
                               "action chunk length the expert denoises over")
        .def_property_readonly("action_dim", &PySmolVLA::action_dim,
                               "padded action dimension (output width)")
        .def_property_readonly("state_dim", &PySmolVLA::state_dim,
                               "padded state dimension the state input is zero-filled to")
        .def("__repr__", [](const PySmolVLA& s) {
            return "<smolvla_npu.SmolVLA path='" + s.model_path() + "' chunk_size="
                 + std::to_string(s.chunk_size()) + " action_dim="
                 + std::to_string(s.action_dim()) + ">";
        });

    m.def("seed", [](unsigned int s) { srand(s); }, py::arg("seed"),
          "Seed the global RNG that forward() samples noise from when no explicit\n"
          "noise tensor is given. Matches run.exe's srand(0).");

    m.attr("MAX_IMAGES")      = smolvla_cfg::MAX_IMAGES;
    m.attr("IMAGE_SIZE")      = smolvla_cfg::IMAGE_SIZE;
    m.attr("MAX_PREFIX_LEN")  = smolvla_cfg::MAX_PREFIX_LEN;

    report_openmp();
}

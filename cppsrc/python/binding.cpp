#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "core/tensor.hpp"
#include "core/types.hpp"
#include "ops/elementwise.hpp"
#include "ops/matmul.hpp"
#include "ops/softmax.hpp"
#include "ops/layernorm.hpp"

namespace py = pybind11;
using namespace vlm;
using gil_release = py::call_guard<py::gil_scoped_release>;

namespace {

DType numpy_to_dtype(const py::dtype& dt) {
    if (dt.is(py::dtype::of<float>())) return DType::Fp32;
    if (dt.kind() == 'f' && dt.itemsize() == 2) return DType::Fp16;
    throw std::runtime_error("unsupported numpy dtype; expected float32 or float16");
}

py::dtype dtype_to_numpy(DType d) {
    switch (d) {
        case DType::Fp32: return py::dtype::of<float>();
        case DType::Fp16: return py::dtype("float16");
        case DType::Bf16: throw std::runtime_error("bf16 has no numpy equivalent");
    }
    throw std::runtime_error("unknown dtype");
}

std::shared_ptr<Tensor> from_numpy(py::array arr, bool requires_grad) {
    auto dt = numpy_to_dtype(arr.dtype());
    py::array contig = py::array::ensure(arr, py::array::c_style);
    std::vector<int64_t> shape(contig.shape(), contig.shape() + contig.ndim());
    auto t = std::make_shared<Tensor>(Tensor::empty(shape, dt, Device::CPU));
    t->set_requires_grad(requires_grad);
    std::memcpy(t->data(), contig.data(), t->nbytes());
    return t;
}

py::array to_numpy(const Tensor& t) {
    if (t.device != Device::CPU) {
        throw std::runtime_error("to_numpy requires CPU tensor; call .to(Device.CPU) first");
    }
    std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());
    py::array out(dtype_to_numpy(t.dtype), shape);
    std::memcpy(out.mutable_data(), t.data(), t.nbytes());
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    py::enum_<DType>(m, "DType")
        .value("Fp32", DType::Fp32)
        .value("Fp16", DType::Fp16)
        .value("Bf16", DType::Bf16);

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA);

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def_static("empty", &Tensor::empty,
                    py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU,
                    gil_release())
        .def_static("zeros", &Tensor::zeros,
                    py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU,
                    gil_release())
        .def_static("ones", &Tensor::ones,
                    py::arg("shape"), py::arg("dtype"), py::arg("device") = Device::CPU,
                    gil_release())
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("dtype", &Tensor::dtype)
        .def_readonly("device", &Tensor::device)
        .def_property("requires_grad",
                      [](const Tensor& t) { return t.requires_grad; },
                      [](Tensor& t, bool r) { t.set_requires_grad(r); })
        .def_property_readonly("is_leaf", &Tensor::is_leaf)
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("grad", &Tensor::grad)
        .def("to", &Tensor::to, py::arg("device"), gil_release())
        .def("zero_grad", &Tensor::zero_grad, gil_release())
        .def("backward", py::overload_cast<>(&Tensor::backward), gil_release())
        .def("backward", py::overload_cast<const Tensor&>(&Tensor::backward),
             py::arg("grad_output"), gil_release())
        .def("__repr__", [](const Tensor& t) {
            static const char* dtype_names[] = {"Fp32", "Fp16", "Bf16"};
            std::ostringstream s;
            s << "Tensor(shape=[";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i) s << ", ";
                s << t.shape[i];
            }
            s << "], dtype=" << dtype_names[static_cast<int>(t.dtype)]
              << ", device=" << (t.device == Device::CPU ? "CPU" : "CUDA") << ")";
            return s.str();
        });

    m.def("from_numpy", &from_numpy, py::arg("array"), py::arg("requires_grad") = false);
    m.def("to_numpy", &to_numpy, py::arg("tensor"));

    m.def("add", &add, gil_release());
    m.def("mul", &mul, gil_release());
    m.def("relu", &relu, gil_release());
    m.def("sum_all", &sum_all, gil_release());
    m.def("matmul", &matmul, gil_release());
    m.def("matmul_a_bt", &matmul_a_bt, gil_release());
    m.def("matmul_at_b", &matmul_at_b, gil_release());
    m.def("softmax", &softmax, gil_release());
    m.def("layernorm", &layernorm,
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
          gil_release());
    m.def("scaled_add_inplace", &scaled_add_inplace,
          py::arg("dst"), py::arg("src"), py::arg("alpha"),
          gil_release());
}

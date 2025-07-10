#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Spin.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Spin, m)
{
    py::class_<Spin>(m, "Spin")
        .def(py::init<int>(), py::arg("L"))
        .def("set_parameters", &Spin::set_parameters,
             py::arg("set_temp"), py::arg("set_J11"), py::arg("set_J12"),
             py::arg("set_J21"), py::arg("set_J22"), py::arg("set_K"))
        .def("get_parameters", &Spin::get_parameters)

        // 原始 set_record 方法
        .def("set_record", &Spin::set_record, py::arg("new_record"))

        // 新增：从 numpy 数组设置记录
        .def("set_record_from_array", [](Spin &self, py::array_t<float> arr)
             {
            if (arr.ndim() != 4 || arr.shape(3) != 3) {
                throw std::runtime_error("Array must have shape (num_saving, L, L, 3)");
            }
            
            int num_saving = arr.shape(0);
            int L = arr.shape(1);
            int expected_spins = L * L;
            
            if (arr.shape(2) != L) {
                throw std::runtime_error("Array dimensions must be square (L x L)");
            }
            
            auto r = arr.unchecked<4>();
            std::vector<std::vector<std::array<float, 3>>> new_record;
            new_record.reserve(num_saving);
            
            for (int n = 0; n < num_saving; ++n) {
                std::vector<std::array<float, 3>> frame;
                frame.reserve(expected_spins);
                
                for (int i = 0; i < L; ++i) {
                    for (int j = 0; j < L; ++j) {
                        frame.push_back({
                            r(n, i, j, 0),
                            r(n, i, j, 1),
                            r(n, i, j, 2)
                        });
                    }
                }
                new_record.push_back(std::move(frame));
            }
            
            self.set_record(new_record); })

        .def("run", &Spin::run, py::arg("step"), py::arg("spacing"))

        // 新增：获取 saving 为 numpy 数组
        .def("get_saving", [](const Spin &self) {
            auto saving = self.get_saving();
            int num_saving = saving.size();
            if (num_saving == 0) {
                throw std::runtime_error("No saving records available");
            }
            
            int total_spins = saving[0].size();
            int L = static_cast<int>(std::sqrt(total_spins));
            
            if (L * L != total_spins) {
                throw std::runtime_error("Invalid spin configuration: not a square lattice");
            }
            
            // 创建 (num_saving, L, L, 3) 形状的数组
            auto result = py::array_t<float>({num_saving, L, L, 3});
            auto r = result.mutable_unchecked<4>();
            
            for (int n = 0; n < num_saving; ++n) {
                if (saving[n].size() != static_cast<size_t>(total_spins)) {
                    throw std::runtime_error("Inconsistent spin count in saving records");
                }
                for (int i = 0; i < L; ++i) {
                    for (int j = 0; j < L; ++j) {
                        int idx = i * L + j;
                        r(n, i, j, 0) = saving[n][idx][0];
                        r(n, i, j, 1) = saving[n][idx][1];
                        r(n, i, j, 2) = saving[n][idx][2];
                    }
                }
            }
            return result; })

        .def(py::pickle(
            [](const Spin &spin) { 
                // __getstate__
                return py::make_tuple(
                    spin.get_parameters(),
                    spin.get_saving());
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid state!");
                }

                auto saving = t[1].cast<std::vector<std::vector<std::array<float, 3>>>>();
                int L = saving.empty() ? 1 : static_cast<int>(std::sqrt(saving[0].size()));

                Spin spin(L);

                auto params = t[0].cast<std::tuple<float, float, float, float, float, float>>();
                spin.set_parameters(
                    std::get<0>(params),
                    std::get<1>(params),
                    std::get<2>(params),
                    std::get<3>(params),
                    std::get<4>(params),
                    std::get<5>(params));

                if (!saving.empty())
                {
                    spin.set_record(saving);
                }
                return spin;
            }));
}
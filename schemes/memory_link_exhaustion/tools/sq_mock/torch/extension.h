// minimal stand-in for syntax checks / mock runs (no torch locally)
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
namespace pybind11 {
struct gil_scoped_release {};
struct object {};
struct bytes : object { bytes(const std::string&) {} };
struct tuple : object {};
template <class... A> tuple make_tuple(A&&...) { return {}; }
}

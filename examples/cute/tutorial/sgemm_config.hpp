#pragma once
#include <cute/layout.hpp>

using namespace cute;
namespace ParamNT {
    const static int bM = 128;
    const static int bN = 128;
    const static int bK = 16;
    const static int bP = 4;
    const static auto warp_layout = make_layout(make_shape(_2{}, _4{}));
    using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
}

namespace ParamTN {
    const static int bM = 128;
    const static int bN = 128;
    const static int bK = 16;
    const static int bP = 4;
    const static auto warp_layout = make_layout(make_shape(_2{}, _4{}));
    using mma_atom = SM80_16x8x8_F16F16F16F16_TN;
}
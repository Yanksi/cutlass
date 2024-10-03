#include <cute/tensor.hpp>
#include <iostream>
#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>

using namespace cute;

int main() {
    // IntTuple tuple = {1, 2, 3};
    // Copy_Atom atom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half>{};
    // // TiledMMA mma = make_tiled_mma(SM80_16x8x8_F32F16F16F32_TN{},
    // //                               Layout<Shape <_1, _2>,
    // //                                      Stride<_2, _1>>{}
    // //                             //   Tile<
    // //                             //     _16,
    // //                             //     _16,
    // //                             //     Layout<Shape<_2, _4, _2>,
    // //                             //            Stride<_1, _4, _2>
    // //                             //     >>{}
    // //                                      );   // 2x2 n-major layout of Atoms
    // TiledCopy copyA = make_tiled_copy(atom,
    //                                 make_layout(Shape<_128, _2>{}, LayoutRight{}),
    //                                 Layout<Shape< _1, _8>>{});// Val layout  4x1 m-major
    // // ThrMMA thr_mma = mma.get_slice(0);
    // print_latex(copyA);
    TiledMMA mmaC = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
                                 Layout<Shape<_2,_4>>{});  // 16x8x8 TiledMMA
    float smemA[1024 * 8];

    auto sA_atom = make_layout(make_shape (_128{}, _32{}), LayoutRight{}); // (m,k) -> smem_idx; padded k-major
    auto sA_layout = tile_to_shape(sA_atom, make_shape(_128{}, _32{}, _4{}));
    Tensor sA = make_tensor(make_gmem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K,PIPE)

    auto mma_k = tile_size<2>(mmaC);
    Tensor sA_p = logical_divide(sA, make_tile(_, make_layout(mma_k), _));  // (BLK_M,(mma_k, BLK_mma_K),PIPE)
    print(size<1,1>(sA_p));
    // print(sA_p.shape());
    // Layout original_layout = make_layout(Shape<_32, _32>{});
    // auto divided = logical_divide(original_layout, make_shape(_, _8{}));
    // print(divided);
    // print(coalesce(divided, Step<_1, Step<_1, _1>>{}));
    // print(size(mma));
    // print(rank(nested_shape));
    // print(depth(nested_shape));
}
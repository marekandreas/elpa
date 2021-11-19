/* This file might be a modified version of the original implementation kindly
 * provided by NVIDIA under the MIT License. The unmodified version can be found
 * in the src at src/shipped_srcs/NVIDIA_A100_kernel/
 *
 * Nov 2021, A. Marek, MPCDF
 *
 */

/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 NVIDIA CORPORATION
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

constexpr int pad = 0;

// How big a tile each warp possess?
constexpr int WARP_M = 1;
constexpr int WARP_N = 1;

// We use the m8n8k4 DMMA instruction
constexpr int INST_M = 8;
constexpr int INST_N = 8;
constexpr int INST_K = 4;

constexpr int MMA_M = INST_M * WARP_M;
constexpr int MMA_N = INST_N * WARP_N;
constexpr int MMA_K = INST_K;

template<int bK, int bN>
__device__ inline int shared_memory_offset(int k, int n) {
  // Shared memory layout for MMA version. The pad of `4` is used to get rid shared memory
  // bank conflicts.
  return k + (bK + 4) * n;
}

__device__ inline constexpr int shared_memory_bytes(int bK, int bN) {
  // Shared memory size for the bM by bK matrix. Version for the MMA.
  return bN * (bK + 4);
}

struct WarpRegisterMapping {
  int lane_id;
  int group_id;
  int thread_id_in_group;

  __device__ WarpRegisterMapping(int thread_id) :
    lane_id(thread_id & 31),
    group_id(lane_id >> 2),
    thread_id_in_group(lane_id & 3)
  {
  }
};

struct MmaOperandA {

  using reg_type = double;
  reg_type reg = 0;

  __device__ inline void construct_sum(void *smem, int tile_k, const WarpRegisterMapping &wrm)
  { // Assuming col major smem layout

    reg_type *A = reinterpret_cast<reg_type *>(smem);
    int k = tile_k * MMA_K + wrm.thread_id_in_group;
    int m = wrm.group_id;
    reg = m == 0 ? A[k] : 0;
  }

};

struct MmaOperandB {

  using reg_type = double;
  reg_type reg = 0;

  template <int bK, int bN> __device__ inline void load(void *smem, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
  {
    reg_type *B = reinterpret_cast<reg_type *>(smem);
    int k = tile_k * MMA_K + wrm.thread_id_in_group;
    int n = tile_n * MMA_N + wrm.group_id;
    reg = B[shared_memory_offset<bK, bN>(k, n)];
  }
};

struct MmaOperandC {

  using reg_type = double;
  reg_type reg[2];

  __device__ MmaOperandC()
  {
#pragma unroll
    for (int i = 0; i < WARP_M * WARP_N * 2; i++) { reg[i] = 0; }
  }

  __device__ void store_sum(void *smem, int tile_n, const WarpRegisterMapping &wrm)
  {
    reg_type *C = reinterpret_cast<reg_type *>(smem);
    const int m = wrm.group_id;
    const int n = tile_n * MMA_N + wrm.thread_id_in_group * 2;

    if (m == 0) {
      C[n + 0] = reg[0];
      C[n + 1] = reg[1];
    }
  }
};

template <int bK, int bN, int total_warp> // bM == 1
__device__ void sum(double *smem_a, double *smem_b, double *smem_c) {

  constexpr int tile_row_dim = 1;          // number of tiles in the col dimension
  constexpr int tile_col_dim = bN / MMA_N; // number of tiles in the row dimension
  constexpr int tile_acc_dim = bK / MMA_K; // number of tiles in the acc dimension

	constexpr int total_tile = tile_row_dim * tile_col_dim;
	constexpr int warp_cycle = total_tile / total_warp;

	static_assert(total_tile % total_warp == 0, "Total number of tiles should be divisible by the number of warps.");

	const int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	const int warp_id = thread_id / 32;

	WarpRegisterMapping wrm(thread_id);

#pragma unroll
	for (int c = 0; c < warp_cycle; c++) {

		MmaOperandC op_c;

		// The logical warp assigned to each part of the matrix.
		int logical_warp_index = warp_id * warp_cycle + c;
		int tile_n = logical_warp_index;

#pragma unroll
		for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
			MmaOperandA op_a;
			op_a.construct_sum(smem_a, tile_k, wrm);

			MmaOperandB op_b;
			op_b.template load<bK, bN>(smem_b, tile_k, tile_n, wrm);

			asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
					: "+d"(op_c.reg[0]), "+d"(op_c.reg[1]) : "d"(op_a.reg), "d"(op_b.reg)
					);
		}

		op_c.store_sum(smem_c, tile_n * MMA_N, wrm);
	}

}

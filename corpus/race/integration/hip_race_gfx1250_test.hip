// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_race_gfx1250_test.hip
/// @brief gfx1250 HIP race detector tests: GFX12 split wait counters and tensor
///        DMA (tensor_load_to_lds, tensor_store_from_lds).
///
/// A tensor load writes LDS and a tensor store reads it, asynchronously, until
/// s_wait_tensorcnt retires the transfer. Transfers from one wave complete in
/// issue order and independently of DS operations, but may access LDS out of
/// order, so overlapping transfers from one wave also need a TENSORcnt wait
/// between them. Other waves see their LDS effects only after the issuing wave
/// waits and the workgroup passes a barrier. Each race case is its clean
/// partner with one step removed. Clean cases also check the data moved; raced
/// values are unconstrained.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_test_support.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  const hipError_t reset_status = hipDeviceReset();
  if (reset_status != hipSuccess) {
    std::fprintf(stderr, "hipDeviceReset failed: %s\n", hipGetErrorString(reset_status));
    return 1;
  }
  return rc;
}

namespace {

using rocjitsu::test::FindingCount;
using rocjitsu::test::RaceExpectation;

constexpr int kWave = 32; // gfx1250 runs wave32.
/// The ints in one tile: a single row with one int per lane. The tests' tensors
/// are this wide too, unless a case narrows them.
constexpr int kTileElements = kWave;
constexpr int kTileBytes = kTileElements * static_cast<int>(sizeof(int));
/// A value no transfer in these tests produces. Memory filled with it shows
/// whether a transfer wrote there.
constexpr int kUnwrittenValue = -7;

using Gfx1250RaceTest = rocjitsu::test::RaceTestBase;

/// `count` consecutive ints starting at `first`.
std::vector<int> range(int first, int count = kTileElements) {
  std::vector<int> values(count);
  for (int index = 0; index < count; ++index)
    values[index] = first + index;
  return values;
}

// --- Tensor DMA descriptors ---

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

/// Descriptor groups 0 and 1 (CDNA5 ISA 10.11.4, Tables 62 and 63). Kernels take them
/// by value, so they arrive in SGPRs; groups 2 and 3 are NULL.
struct TensorDesc {
  v4i group0;
  v8i group1;
};

/// A one-row transfer of 4-byte elements between global memory and LDS.
struct Tile {
  const void *global = nullptr;
  uint32_t lds_offset = 0;
  uint32_t tensor_elements = kTileElements; ///< Elements past this are out of the tensor.
  uint32_t tile_elements = kTileElements;   ///< Zero makes the tile a NOP.
  uint32_t count = 1;                   ///< Zero makes this a NULL tensor.
  bool pad = false;
  uint32_t pad_interval = 0; ///< 2 << pad_interval dwords between pads.
  uint32_t pad_amount = 0;   ///< pad_amount + 1 dwords of padding.
};

TensorDesc describe(const Tile &tile) {
  uint32_t words[12] = {};
  const uint64_t address = reinterpret_cast<uint64_t>(tile.global);
  words[0] = tile.count;
  words[1] = tile.lds_offset;
  words[2] = static_cast<uint32_t>(address);
  words[3] = static_cast<uint32_t>((address >> 32) & 0x1ffffffu) | (2u << 30); // type 2.
  words[4] = (2u << 16) | (uint32_t{tile.pad} << 20) | (tile.pad_interval << 22) |
             (tile.pad_amount << 25);
  words[5] = (tile.tensor_elements & 0xffffu) << 16;          // tensor_dim0[15:0].
  words[6] = (tile.tensor_elements >> 16) | (1u << 16);        // tensor_dim1 = 1.
  words[7] = (tile.tile_elements & 0xffffu) << 16;             // tile_dim0.
  words[8] = 1;                                                // tile_dim1.
  words[9] = kTileElements;                                        // tensor_dim0_stride.
  TensorDesc desc;
  std::memcpy(&desc.group0, &words[0], sizeof(desc.group0));
  std::memcpy(&desc.group1, &words[4], sizeof(desc.group1));
  return desc;
}

// --- Device helpers. Every wait is explicit; nothing here lets the compiler
// --- add one. ---

/// This wave's index in the workgroup, as a scalar so branches on it are
/// uniform and the tensor instructions they guard issue once per wave.
__device__ inline int waveIndex() { return __builtin_amdgcn_readfirstlane(threadIdx.x) / kWave; }

__device__ inline int lane() { return threadIdx.x % kWave; }

__device__ inline void tensorLoad(const TensorDesc &desc) {
  asm volatile("tensor_load_to_lds %0, %1" ::"s"(desc.group0), "s"(desc.group1) : "memory");
}

__device__ inline void tensorStore(const TensorDesc &desc) {
  asm volatile("tensor_store_from_lds %0, %1" ::"s"(desc.group0), "s"(desc.group1) : "memory");
}

template <int Count> __device__ inline void waitTensorcnt() {
  asm volatile("s_wait_tensorcnt %0" ::"n"(Count) : "memory");
}

__device__ inline void waitDscnt() { asm volatile("s_wait_dscnt 0" ::: "memory"); }

/// A workgroup barrier with no implied waits, unlike __syncthreads().
__device__ inline void barrier() {
  asm volatile("s_barrier_signal -1\n"
               "s_barrier_wait -1" ::: "memory");
}

/// Read an int from LDS and wait for it, so a case reports only the LDS race
/// it is about and never a race on the destination register.
__device__ inline int ldsReadAndWait(uint32_t byte_offset) {
  int value;
  asm volatile("ds_load_b32 %0, %1\n"
               "s_wait_dscnt 0"
               : "=v"(value)
               : "v"(byte_offset)
               : "memory");
  return value;
}

/// Issue an LDS write without waiting for it to complete.
__device__ inline void ldsWrite(uint32_t byte_offset, int value) {
  asm volatile("ds_store_b32 %0, %1" ::"v"(byte_offset), "v"(value) : "memory");
}

/// Fill one tile of LDS with `value` and wait for the writes.
__device__ inline void ldsFillAndWait(uint32_t byte_offset, int value) {
  ldsWrite(byte_offset + lane() * 4, value);
  waitDscnt();
}

static constexpr RaceExpectation kTensorLoadThenReadRace{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .access = "read",
    .producer = "tensor_load_to_lds",
    .consumer = "ds_load_b32",
};

static constexpr RaceExpectation kTensorStoreThenWriteRace{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .access = "write",
    .producer = "tensor_store_from_lds",
    .consumer = "ds_store_b32",
};

static constexpr RaceExpectation kDsWriteThenTensorStoreRace{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .access = "read",
    .producer = "ds_store_b32",
    .consumer = "tensor_store_from_lds",
};

static constexpr RaceExpectation kTensorLoadThenTensorStoreRace{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .access = "read",
    .producer = "tensor_load_to_lds",
    .consumer = "tensor_store_from_lds",
};

static constexpr RaceExpectation kTensorStoreThenTensorLoadRace{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .access = "write",
    .producer = "tensor_store_from_lds",
    .consumer = "tensor_load_to_lds",
};

// --- GFX12 split wait counters ---

template <bool Wait> __global__ void vgpr_loadcnt_kernel(const int *src, int *dst) {
  int value;
  if (Wait)
    asm volatile("global_load_b32 %0, %1, off\n"
                 "s_wait_loadcnt 0"
                 : "=v"(value)
                 : "v"(&src[threadIdx.x]));
  else
    asm volatile("global_load_b32 %0, %1, off\n"
                 "s_nop 0"
                 : "=v"(value)
                 : "v"(&src[threadIdx.x]));
  dst[threadIdx.x] = value;
}

TEST_F(Gfx1250RaceTest, vgpr_loadcnt) {
  auto *src = allocWithData<int>(kWave);
  auto *dst = alloc<int>(kWave);
  vgpr_loadcnt_kernel<true><<<1, kWave>>>(src, dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0, kWave));
}

TEST_F(Gfx1250RaceTest, vgpr_loadcnt_race) {
  auto *src = allocWithData<int>(kWave);
  auto *dst = alloc<int>(kWave);
  vgpr_loadcnt_kernel<false><<<1, kWave>>>(src, dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace({.findings = FindingCount::OneOrMore,
              .type = "VGPR",
              .producer = "global_load_b32",
              .consumer = "global_store_b32"});
}

// --- The issuing wave must wait for TENSORCNT, and only TENSORCNT ---

enum class Wait { Tensorcnt, None, OtherCounters };

template <Wait W> __global__ void tensor_load_same_wave_kernel(TensorDesc desc, int *dst) {
  tensorLoad(desc);
  if (W == Wait::Tensorcnt)
    waitTensorcnt<0>();
  if (W == Wait::OtherCounters)
    asm volatile("s_wait_loadcnt 0\n"
                 "s_wait_dscnt 0" ::: "memory");
  dst[threadIdx.x] = ldsReadAndWait(threadIdx.x * 4);
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_load_tensorcnt) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_same_wave_kernel<Wait::Tensorcnt><<<1, kWave, kTileBytes>>>(describe({.global = src}),
                                                                          dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_load_missing_tensorcnt_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_same_wave_kernel<Wait::None><<<1, kWave, kTileBytes>>>(describe({.global = src}),
                                                                     dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// LOADcnt and DScnt waits do not cover tensor DMA.
TEST_F(Gfx1250RaceTest, tensor_load_other_counters_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_same_wave_kernel<Wait::OtherCounters>
      <<<1, kWave, kTileBytes>>>(describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// --- A store must omit the holes between partially in-bounds tile rows ---

TensorDesc describeStoreWithHoles(void *global) {
  auto desc = describe({.global = global, .tensor_elements = 16, .tile_elements = 32});
  desc.group1[2] = 2u << 16; // tensor_dim1 = 2.
  desc.group1[4] = 2;        // tile_dim1 = 2.
  desc.group1[5] = 16;       // Contiguous 16-element global rows.
  return desc;
}

template <bool WriteHole>
__global__ void tensor_store_holes_kernel(TensorDesc store) {
  ldsFillAndWait(0, 100 + lane());
  ldsFillAndWait(kTileBytes, 200 + lane());
  tensorStore(store);
  if (lane() < 16)
    ldsWrite((WriteHole ? 64 : 0) + lane() * 4, 300 + lane());
  waitDscnt();
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_holes) {
  auto *stored = alloc<int>(kTileElements);
  tensor_store_holes_kernel<true><<<1, kWave, 2 * kTileBytes>>>(describeStoreWithHoles(stored));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  auto expected = range(100, 16);
  const auto second = range(200, 16);
  expected.insert(expected.end(), second.begin(), second.end());
  ExpectValues(stored, expected);
}

TEST_F(Gfx1250RaceTest, tensor_store_holes_adjacent_data_race) {
  auto *stored = alloc<int>(kTileElements);
  tensor_store_holes_kernel<false><<<1, kWave, 2 * kTileBytes>>>(describeStoreWithHoles(stored));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenWriteRace);
}

// --- Other waves need the issuing wave's wait and a barrier ---

enum class Sync { WaitAndBarrier, WaitOnly, BarrierOnly };

// Wave 0 loads the tile; wave 1 reads it.
template <Sync S>
__global__ __attribute__((amdgpu_flat_work_group_size(2 * kWave, 2 * kWave))) void
tensor_load_cross_wave_kernel(TensorDesc desc, int *dst) {
  const int wave = waveIndex();
  if (wave == 0) {
    tensorLoad(desc);
    if (S != Sync::BarrierOnly)
      waitTensorcnt<0>();
  }
  if (S != Sync::WaitOnly)
    barrier();
  if (wave == 1)
    dst[lane()] = ldsReadAndWait(lane() * 4);
  if (wave == 0)
    waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_load_cross_wave) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_cross_wave_kernel<Sync::WaitAndBarrier>
      <<<1, 2 * kWave, kTileBytes>>>(describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_load_cross_wave_no_barrier_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_cross_wave_kernel<Sync::WaitOnly>
      <<<1, 2 * kWave, kTileBytes>>>(describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  const auto parsed = rocjitsu::test::parseRaceLogFromEnvironment();
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  // Either wave may issue first. Require the same conflicting instructions
  // whether the read follows the tensor load or the tensor load follows the read.
  const auto load_then_read =
      rocjitsu::test::matchRaceExpectation(parsed.records, kTensorLoadThenReadRace);
  const auto read_then_load = rocjitsu::test::matchRaceExpectation(
      parsed.records, {.findings = FindingCount::OneOrMore,
                       .type = "LDS",
                       .access = "write",
                       .producer = "ds_load_b32",
                       .consumer = "tensor_load_to_lds"});
  EXPECT_TRUE(load_then_read.ok() || read_then_load.ok())
      << "Tensor load before LDS read: " << load_then_read.message()
      << "\nLDS read before tensor load: " << read_then_load.message();
}

// A barrier retires only transfers their wave has already waited for.
TEST_F(Gfx1250RaceTest, tensor_load_cross_wave_no_tensorcnt_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_cross_wave_kernel<Sync::BarrierOnly>
      <<<1, 2 * kWave, kTileBytes>>>(describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// --- Double-buffered pipeline ---

// Wave 0 loads two tiles, waits for all but the newest, and the workgroup
// synchronizes. The older tile is ready for wave 1; the newer one is not.
template <uint32_t ReadOffset>
__global__ __attribute__((amdgpu_flat_work_group_size(2 * kWave, 2 * kWave))) void
tensor_load_pipeline_kernel(TensorDesc older, TensorDesc newer, int *dst) {
  const int wave = waveIndex();
  if (wave == 0) {
    tensorLoad(older);
    tensorLoad(newer);
    waitTensorcnt<1>();
  }
  barrier();
  if (wave == 1)
    dst[lane()] = ldsReadAndWait(ReadOffset + lane() * 4);
  if (wave == 0)
    waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_load_pipeline_oldest) {
  auto *older = allocWithData<int>(kTileElements);
  auto *newer = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_pipeline_kernel<0><<<1, 2 * kWave, 2 * kTileBytes>>>(
      describe({.global = older}), describe({.global = newer, .lds_offset = kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_load_pipeline_newest_race) {
  auto *older = allocWithData<int>(kTileElements);
  auto *newer = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_pipeline_kernel<kTileBytes><<<1, 2 * kWave, 2 * kTileBytes>>>(
      describe({.global = older}), describe({.global = newer, .lds_offset = kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// Half the lanes read the completed tile; half read the pending tile.
TEST_F(Gfx1250RaceTest, tensor_load_pipeline_straddling_race) {
  auto *older = allocWithData<int>(kTileElements);
  auto *newer = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_pipeline_kernel<kTileBytes / 2><<<1, 2 * kWave, 2 * kTileBytes>>>(
      describe({.global = older}), describe({.global = newer, .lds_offset = kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
  const auto parsed = rocjitsu::test::parseRaceLogFromEnvironment();
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  for (const auto &record : parsed.records) {
    EXPECT_EQ(record.type, "LDS");
    EXPECT_EQ(record.access, "read");
    EXPECT_EQ(record.wave, 1);
    EXPECT_GE(record.lane, kWave / 2);
    EXPECT_LT(record.lane, kWave);
    EXPECT_GE(record.reg, kTileBytes);
    EXPECT_LT(record.reg, kTileBytes + kTileBytes / 2);
  }
}

template <bool WaitForAsync>
__global__ void async_load_then_tensor_store_kernel(const int *src, TensorDesc store) {
  asm volatile("global_load_async_to_lds_b32 %[lds], %[offset], %[base]\n"
               :
               : [lds] "v"(lane() * 4), [offset] "v"(lane() * 4), [base] "s"(src)
               : "memory");
  if constexpr (WaitForAsync)
    asm volatile("s_wait_asynccnt 0" ::: "memory");
  else
    // Waiting for tensor DMA must not complete an ordinary async LDS load.
    waitTensorcnt<0>();
  tensorStore(store);
  waitTensorcnt<0>();
  asm volatile("s_wait_asynccnt 0" ::: "memory");
}

TEST_F(Gfx1250RaceTest, async_load_then_tensor_store) {
  auto *src = allocWithData<int>(kTileElements), *stored = alloc<int>(kTileElements);
  async_load_then_tensor_store_kernel<true><<<1, kWave, kTileBytes>>>(
      src, describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(0));
}

TEST_F(Gfx1250RaceTest, async_load_then_tensor_store_wrong_wait_race) {
  auto *src = allocWithData<int>(kTileElements), *stored = alloc<int>(kTileElements);
  async_load_then_tensor_store_kernel<false><<<1, kWave, kTileBytes>>>(
      src, describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace({.findings = FindingCount::OneOrMore,
              .type = "LDS",
              .access = "read",
              .producer = "global_load_async_to_lds_b32",
              .consumer = "tensor_store_from_lds"});
}

// --- A tensor store's LDS source ---

// Wave 0 stores the tile; wave 1 then overwrites it in LDS. The barrier orders
// the overwrite after the store issues. Without the wait, it does not retire
// the store, so the overwrite races with it.
template <bool Synchronized>
__global__ __attribute__((amdgpu_flat_work_group_size(2 * kWave, 2 * kWave))) void
tensor_store_source_overwrite_kernel(TensorDesc desc) {
  const int wave = waveIndex();
  if (wave == 0) {
    ldsFillAndWait(0, 100 + lane());
    tensorStore(desc);
    if (Synchronized)
      waitTensorcnt<0>();
  }
  barrier();
  if (wave == 1)
    ldsFillAndWait(0, -1);
  if (wave == 0)
    waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_source_overwrite) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_source_overwrite_kernel<true>
      <<<1, 2 * kWave, kTileBytes>>>(describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(100));
}

TEST_F(Gfx1250RaceTest, tensor_store_source_overwrite_race) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_source_overwrite_kernel<false>
      <<<1, 2 * kWave, kTileBytes>>>(describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenWriteRace);
}

// DS writes and tensor DMA complete independently, so storing a tile the wave
// just wrote takes a DScnt wait first.
template <bool WaitDscnt, uint32_t WriteOffset = 0, int WriteElements = kTileElements>
__global__ void tensor_store_after_ds_write_kernel(TensorDesc desc) {
  if constexpr (WriteElements < kTileElements)
    ldsFillAndWait(0, kUnwrittenValue);
  if (lane() < WriteElements)
    ldsWrite(WriteOffset + lane() * 4, 100 + lane());
  if (WaitDscnt)
    waitDscnt();
  tensorStore(desc);
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_dscnt) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_after_ds_write_kernel<true><<<1, kWave, kTileBytes>>>(describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(100));
}

TEST_F(Gfx1250RaceTest, tensor_store_missing_dscnt_race) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_after_ds_write_kernel<false><<<1, kWave, kTileBytes>>>(describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kDsWriteThenTensorStoreRace);
}

// Exercise strict containment in both directions at an interior LDS word.
TEST_F(Gfx1250RaceTest, tensor_store_ds_subset) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_after_ds_write_kernel<true, 60, 1><<<1, kWave, kTileBytes>>>(
      describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  std::vector<int> expected(kTileElements, kUnwrittenValue);
  expected[15] = 100;
  ExpectValues(dst, expected);
}

TEST_F(Gfx1250RaceTest, tensor_store_ds_subset_race) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_after_ds_write_kernel<false, 60, 1><<<1, kWave, kTileBytes>>>(
      describe({.global = dst}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kDsWriteThenTensorStoreRace);
}

TEST_F(Gfx1250RaceTest, tensor_store_tensor_subset) {
  auto *dst = alloc<int>(1);
  tensor_store_after_ds_write_kernel<true><<<1, kWave, kTileBytes>>>(
      describe({.global = dst, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, std::vector<int>{115});
}

TEST_F(Gfx1250RaceTest, tensor_store_tensor_subset_race) {
  auto *dst = alloc<int>(1);
  tensor_store_after_ds_write_kernel<false><<<1, kWave, kTileBytes>>>(
      describe({.global = dst, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kDsWriteThenTensorStoreRace);
}

// --- Same-wave tensor loads and stores over one tile need a TENSORcnt wait ---

// Tensor transfers from one wave complete on TENSORcnt in issue order (CDNA5 ISA
// 10.11.1), but that orders completion, not LDS access. A tensor load moves its
// data with async loads (10.11.3), and async loads and stores may access LDS out
// of order, even relative to each other (10.8). The detector therefore reports a
// tensor load and a tensor store from the same wave that touch the same bytes,
// in either order, until a TENSORcnt wait retires the earlier one.

// A load into the tile an earlier store is still reading may overwrite bytes the
// store has not read yet.
template <bool WaitTensorcnt, uint32_t ReadOffset = 0, int ReadElements = kTileElements,
          int LdsElements = kTileElements>
__global__ void tensor_store_then_load_same_wave_kernel(TensorDesc store, TensorDesc load,
                                                        int *dst) {
  for (uint32_t offset = 0; offset < LdsElements * 4; offset += kTileBytes)
    ldsFillAndWait(offset, 100 + offset / 4 + lane());
  tensorStore(store);
  if (WaitTensorcnt)
    waitTensorcnt<0>();
  tensorLoad(load);
  waitTensorcnt<0>();
  if (lane() < ReadElements)
    dst[lane()] = ldsReadAndWait(ReadOffset + lane() * 4);
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_same_wave) {
  auto *stored = alloc<int>(kTileElements);
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_store_then_load_same_wave_kernel<true><<<1, kWave, kTileBytes>>>(
      describe({.global = stored}), describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(100));
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_same_wave_race) {
  auto *stored = alloc<int>(kTileElements);
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_store_then_load_same_wave_kernel<false><<<1, kWave, kTileBytes>>>(
      describe({.global = stored}), describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenTensorLoadRace);
}

// Store-source and load-destination footprints need not have equal bounds.
TEST_F(Gfx1250RaceTest, tensor_store_then_load_store_subset) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(1);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<true, 0, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}),
          describe({.global = src, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(115, 1));
  ExpectValues(loaded, range(0, 32));
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_store_subset_race) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(1);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<false, 0, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}),
          describe({.global = src, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenTensorLoadRace);
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_load_subset) {
  auto *src = allocWithData<int>(1);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(1);
  tensor_store_then_load_same_wave_kernel<true, 60, 1, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(100, 32));
  ExpectValues(loaded, range(0, 1));
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_load_subset_race) {
  auto *src = allocWithData<int>(1);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(1);
  tensor_store_then_load_same_wave_kernel<false, 60, 1, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 60, .tensor_elements = 1, .tile_elements = 1}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenTensorLoadRace);
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_overlap_right) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<true, 64, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 64, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(100, 32));
  ExpectValues(loaded, range(0, 32));
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_overlap_right_race) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<false, 64, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 64, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenTensorLoadRace);
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_overlap_left) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<true, 0, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 64, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(116, 32));
  ExpectValues(loaded, range(0, 32));
}

TEST_F(Gfx1250RaceTest, tensor_store_then_load_overlap_left_race) {
  auto *src = allocWithData<int>(32);
  auto *stored = alloc<int>(32);
  auto *loaded = alloc<int>(32);
  tensor_store_then_load_same_wave_kernel<false, 0, 32, 2 * kTileElements>
      <<<1, kWave, 2 * kTileBytes>>>(
          describe({.global = stored, .lds_offset = 64, .tensor_elements = 32, .tile_elements = 32}),
          describe({.global = src, .lds_offset = 0, .tensor_elements = 32, .tile_elements = 32}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenTensorLoadRace);
}

// A store of the tile an earlier load is still writing may read bytes the load
// has not written yet.
template <bool WaitTensorcnt>
__global__ void tensor_load_then_store_same_wave_kernel(TensorDesc load, TensorDesc store) {
  tensorLoad(load);
  if (WaitTensorcnt)
    waitTensorcnt<0>();
  tensorStore(store);
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_load_then_store_same_wave) {
  auto *src = allocWithData<int>(kTileElements);
  auto *stored = allocFilled(kTileElements, kUnwrittenValue);
  tensor_load_then_store_same_wave_kernel<true><<<1, kWave, kTileBytes>>>(
      describe({.global = src}), describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  // The store copies the tile the load wrote.
  ExpectValues(stored, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_load_then_store_same_wave_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *stored = allocFilled(kTileElements, kUnwrittenValue);
  tensor_load_then_store_same_wave_kernel<false><<<1, kWave, kTileBytes>>>(
      describe({.global = src}), describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenTensorStoreRace);
}

template <bool ReadNewer>
__global__ void tensor_mixed_partial_wait_kernel(TensorDesc store, TensorDesc load, int *dst) {
  ldsFillAndWait(0, 100 + lane());
  tensorStore(store);
  tensorLoad(load);
  waitTensorcnt<1>();
  if constexpr (ReadNewer) {
    // The load can still be pending even though the older store is complete.
    dst[lane()] = ldsReadAndWait(kTileBytes + lane() * 4);
  } else {
    // Only the completed store read this tile; the younger load uses another.
    ldsFillAndWait(0, 200 + lane());
  }
  waitTensorcnt<0>();
  if constexpr (!ReadNewer)
    dst[lane()] = ldsReadAndWait(kTileBytes + lane() * 4);
}

TEST_F(Gfx1250RaceTest, tensor_mixed_partial_wait_oldest_store) {
  auto *stored = alloc<int>(kTileElements), *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kTileElements);
  tensor_mixed_partial_wait_kernel<false><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = stored}), describe({.global = src, .lds_offset = kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(100));
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_mixed_partial_wait_newest_load_race) {
  auto *stored = alloc<int>(kTileElements), *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kTileElements);
  tensor_mixed_partial_wait_kernel<true><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = stored}), describe({.global = src, .lds_offset = kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
  ExpectValues(stored, range(100));
}

// --- Tensor instructions that move no data still count on TENSORCNT ---

// A real load, then an instruction that moves nothing, then a wait that lets
// one transfer remain outstanding: that one is the empty instruction, so the
// real load is complete. Pipelines turn a prefetch past the end of a tensor
// into such an instruction.
template <int WaitCount>
__global__ void tensor_load_then_empty_kernel(TensorDesc real, TensorDesc empty, int *loaded,
                                              int *untouched) {
  ldsFillAndWait(kTileBytes, kUnwrittenValue);
  tensorLoad(real);
  tensorLoad(empty);
  waitTensorcnt<WaitCount>();
  loaded[lane()] = ldsReadAndWait(lane() * 4);
  untouched[lane()] = ldsReadAndWait(kTileBytes + lane() * 4);
  waitTensorcnt<0>();
}

// count = 0: a NULL tensor (CDNA5 ISA 10.11.4, Table 62). That a NULL tensor takes a
// TENSORcnt slot is a RocJITsu assumption pending hardware validation; only
// under it does s_wait_tensorcnt 1 retire the real load.
TEST_F(Gfx1250RaceTest, tensor_load_null_tensor_counts) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<1><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}),
      describe({.global = src, .lds_offset = kTileBytes, .count = 0}), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
  ExpectValues(untouched, std::vector<int>(kWave, kUnwrittenValue));
}

// Waiting for two leaves the real load pending whether or not the NULL tensor
// takes a slot.
TEST_F(Gfx1250RaceTest, tensor_load_null_tensor_uncounted_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<2><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}),
      describe({.global = src, .lds_offset = kTileBytes, .count = 0}), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// tile_dim0 = 0: a NOP tile (CDNA5 ISA 10.11.4, Table 63).
TEST_F(Gfx1250RaceTest, tensor_load_empty_tile_counts) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<1><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}),
      describe({.global = src, .lds_offset = kTileBytes, .tile_elements = 0}), loaded,
      untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
  ExpectValues(untouched, std::vector<int>(kWave, kUnwrittenValue));
}

// Empty gathers use the same counter slot as dense NOPs. The two shapes
// exercise zero valid rows and zero row width independently (CDNA5 ISA 10.11.4).
TensorDesc describeEmptyGather(const void *global, bool zero_width) {
  auto desc = describe({.global = global, .lds_offset = kTileBytes,
                        .tile_elements = zero_width ? 0u : uint32_t{kTileElements}});
  desc.group0[0] |= (1u << 30) | (1u << 31);
  desc.group1[4] = zero_width ? 2 : 0;
  return desc;
}

TEST_F(Gfx1250RaceTest, tensor_load_empty_gather_zero_indices_counts) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<1><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}), describeEmptyGather(src, false), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
  ExpectValues(untouched, std::vector<int>(kWave, kUnwrittenValue));
}

TEST_F(Gfx1250RaceTest, tensor_load_empty_gather_zero_indices_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<2><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}), describeEmptyGather(src, false), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

TEST_F(Gfx1250RaceTest, tensor_load_empty_gather_zero_width_counts) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<1><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}), describeEmptyGather(src, true), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
  ExpectValues(untouched, std::vector<int>(kWave, kUnwrittenValue));
}

TEST_F(Gfx1250RaceTest, tensor_load_empty_gather_zero_width_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *untouched = alloc<int>(kWave);
  tensor_load_then_empty_kernel<2><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}), describeEmptyGather(src, true), loaded, untouched);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// A store whose tile lies wholly outside the tensor writes nothing to global
// memory, but still counts. It reads no LDS either, so overwriting its tile
// while it is still pending is safe.
__global__ void tensor_load_then_empty_store_kernel(TensorDesc real, TensorDesc empty,
                                                    int *loaded) {
  tensorLoad(real);
  tensorStore(empty);
  waitTensorcnt<1>();
  loaded[lane()] = ldsReadAndWait(lane() * 4);
  ldsFillAndWait(kTileBytes, 100 + lane());
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_outside_tensor_counts) {
  auto *src = allocWithData<int>(kTileElements);
  auto *untouched = allocFilled(kTileElements, kUnwrittenValue);
  auto *loaded = alloc<int>(kWave);
  tensor_load_then_empty_store_kernel<<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = src}),
      describe({.global = untouched, .lds_offset = kTileBytes, .tensor_elements = 0}), loaded);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
  ExpectValues(untouched, std::vector<int>(kTileElements, kUnwrittenValue));
}

// The real store sits between two empty instructions. Waiting for at most one
// retires the real store; waiting for two may leave it reading its LDS source.
template <int WaitCount>
__global__ void tensor_store_between_empty_stores_kernel(TensorDesc empty, TensorDesc real) {
  ldsFillAndWait(0, 100 + lane());
  ldsFillAndWait(kTileBytes, 200 + lane());
  tensorStore(empty);
  tensorStore(real);
  tensorStore(empty);
  waitTensorcnt<WaitCount>();
  ldsWrite(lane() * 4, -1);
  // An outstanding empty store must not claim to read its nominal LDS source.
  ldsWrite(kTileBytes + lane() * 4, -1);
  waitDscnt();
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_between_empty_stores) {
  auto *stored = alloc<int>(kTileElements);
  auto *untouched = allocFilled(kTileElements, kUnwrittenValue);
  tensor_store_between_empty_stores_kernel<1><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = untouched, .lds_offset = kTileBytes, .tensor_elements = 0}),
      describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(stored, range(100));
  ExpectValues(untouched, std::vector<int>(kTileElements, kUnwrittenValue));
}

TEST_F(Gfx1250RaceTest, tensor_store_between_empty_stores_race) {
  auto *stored = alloc<int>(kTileElements);
  auto *untouched = allocFilled(kTileElements, kUnwrittenValue);
  tensor_store_between_empty_stores_kernel<2><<<1, kWave, 2 * kTileBytes>>>(
      describe({.global = untouched, .lds_offset = kTileBytes, .tensor_elements = 0}),
      describe({.global = stored}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorStoreThenWriteRace);
  ExpectValues(untouched, std::vector<int>(kTileElements, kUnwrittenValue));
}

// --- Elements outside the tensor are zero-filled in LDS ---

constexpr uint32_t kTensorHalf = kTileElements / 2;

// The tile is twice as wide as the tensor; the second half reads as zero
// (CDNA5 ISA 10.11.2). Wave 1 reads the zero-filled half. The kUnwrittenValue
// fill is synchronized first, so it cannot race with that read. The second
// barrier orders the read after the load issues; without the wait, it does not
// retire the load, so the read races with it.
template <bool Synchronized>
__global__ __attribute__((amdgpu_flat_work_group_size(2 * kWave, 2 * kWave))) void
tensor_load_zero_fill_kernel(TensorDesc desc, int *dst) {
  const int wave = waveIndex();
  if (wave == 0)
    ldsFillAndWait(0, kUnwrittenValue);
  barrier();
  if (wave == 0) {
    tensorLoad(desc);
    if (Synchronized)
      waitTensorcnt<0>();
  }
  barrier();
  if (wave == 1)
    dst[lane()] = ldsReadAndWait((kTensorHalf + lane() % kTensorHalf) * 4);
  if (wave == 0)
    waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_load_zero_fill) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_zero_fill_kernel<true><<<1, 2 * kWave, kTileBytes>>>(
      describe({.global = src, .tensor_elements = kTensorHalf}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, std::vector<int>(kWave, 0));
}

TEST_F(Gfx1250RaceTest, tensor_load_zero_fill_cross_wave_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  tensor_load_zero_fill_kernel<false><<<1, 2 * kWave, kTileBytes>>>(
      describe({.global = src, .tensor_elements = kTensorHalf}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// --- LDS padding is not written ---

// 16 dwords of data, then 4 dwords of padding, then the other 16.
constexpr uint32_t kPadOffset = 64;
constexpr uint32_t kPadBytes = 16;

// Wave 1 reads LDS at `ReadOffset` while wave 0's padded load is outstanding.
// The barrier orders the read after the load issues; with no wait before it, it
// does not retire the load.
template <uint32_t ReadOffset>
__global__ __attribute__((amdgpu_flat_work_group_size(2 * kWave, 2 * kWave))) void
tensor_load_padding_kernel(TensorDesc desc, int *loaded, int *concurrent) {
  const int wave = waveIndex();
  if (wave == 0)
    tensorLoad(desc);
  barrier();
  if (wave == 1)
    concurrent[lane()] = ldsReadAndWait(ReadOffset);
  if (wave == 0) {
    waitTensorcnt<0>();
    const uint32_t offset = lane() * 4 + (lane() < 16 ? 0 : kPadBytes);
    loaded[lane()] = ldsReadAndWait(offset);
  }
}

TEST_F(Gfx1250RaceTest, tensor_load_padding_gap) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *concurrent = alloc<int>(kWave);
  tensor_load_padding_kernel<kPadOffset><<<1, 2 * kWave, kTileBytes + kPadBytes>>>(
      describe({.global = src, .pad = true, .pad_interval = 3, .pad_amount = 3}), loaded,
      concurrent);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(loaded, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_load_padding_data_race) {
  auto *src = allocWithData<int>(kTileElements);
  auto *loaded = alloc<int>(kWave);
  auto *concurrent = alloc<int>(kWave);
  tensor_load_padding_kernel<kPadOffset + kPadBytes><<<1, 2 * kWave, kTileBytes + kPadBytes>>>(
      describe({.global = src, .pad = true, .pad_interval = 3, .pad_amount = 3}), loaded,
      concurrent);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// Three padded loads fill interleaved four-dword stripes. A wave may read its
// own completed stripes before peers publish their completion through a barrier.
template <bool ReadPeer>
__global__ __attribute__((amdgpu_flat_work_group_size(3 * kWave, 3 * kWave))) void
tensor_load_interleaved_kernel(TensorDesc first, TensorDesc second, TensorDesc third,
                                int *probes, int *assembled) {
  const int wave = waveIndex();
  if (wave == 0)
    tensorLoad(first);
  else if (wave == 1)
    tensorLoad(second);
  else
    tensorLoad(third);
  // All three write events must exist before any wave probes a stripe.
  barrier();
  waitTensorcnt<0>();
  if (lane() < 12) {
    const int owner = ReadPeer ? (wave + 1) % 3 : wave;
    const uint32_t offset = owner * 16 + (lane() / 4) * 48 + (lane() % 4) * 4;
    probes[wave * 12 + lane()] = ldsReadAndWait(offset);
  }
  barrier();
  if (threadIdx.x < 36)
    assembled[threadIdx.x] = ldsReadAndWait(threadIdx.x * 4);
}

TEST_F(Gfx1250RaceTest, tensor_load_interleaved_own_stripes) {
  auto *first = allocWithData<int>(36);
  auto *second = first + 12;
  auto *third = first + 24;
  auto *probes = alloc<int>(36);
  auto *assembled = alloc<int>(36);
  tensor_load_interleaved_kernel<false><<<1, 3 * kWave, 144>>>(
      describe({.global = first, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}),
      describe({.global = second, .lds_offset = 16, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}),
      describe({.global = third, .lds_offset = 32, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}), probes, assembled);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  std::vector<int> expected_assembled;
  for (int stripe = 0; stripe < 3; ++stripe)
    for (int wave = 0; wave < 3; ++wave)
      for (int element = 0; element < 4; ++element)
        expected_assembled.push_back(wave * 12 + stripe * 4 + element);
  ExpectValues(probes, range(0, 36));
  ExpectValues(assembled, expected_assembled);
}

TEST_F(Gfx1250RaceTest, tensor_load_interleaved_peer_stripes_race) {
  auto *src = allocWithData<int>(12);
  auto *probes = alloc<int>(36);
  auto *assembled = alloc<int>(36);
  tensor_load_interleaved_kernel<true><<<1, 3 * kWave, 144>>>(
      describe({.global = src, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}),
      describe({.global = src, .lds_offset = 16, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}),
      describe({.global = src, .lds_offset = 32, .tensor_elements = 12, .tile_elements = 12,
                .pad = true, .pad_interval = 1, .pad_amount = 7}), probes, assembled);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// --- Iterations and disjoint ranges still belong to one tensor instruction ---

template <bool MultiFirst, int WaitCount, uint32_t Probe>
__global__ void tensor_iteration_count_kernel(TensorDesc multi, TensorDesc single, int *dst) {
  // Two copies of the same row, with a row-sized LDS hole between them.
  const v4i iteration = {0, 2 * kTileElements, 0, 1 << 16};
  const v4i unused = {0, 0, 0, 0};
  if constexpr (!MultiFirst)
    tensorLoad(single);
  asm volatile("tensor_load_to_lds %0, %1, %2, %3"
               :: "s"(multi.group0), "s"(multi.group1), "s"(iteration), "s"(unused) : "memory");
  if constexpr (MultiFirst)
    tensorLoad(single);
  waitTensorcnt<WaitCount>();
  dst[lane()] = ldsReadAndWait(Probe + lane() * 4);
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_iterations_oldest_all_ranges) {
  auto *src = allocWithData<int>(kTileElements), *dst = alloc<int>(kTileElements);
  auto multi = describe({.global = src});
  multi.group1[0] |= 1u << 19; // Enable the iteration descriptor.
  tensor_iteration_count_kernel<true, 1, 2 * kTileBytes>
      <<<1, kWave, 5 * kTileBytes>>>(
          multi, describe({.global = src, .lds_offset = 4 * kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0));
}

TEST_F(Gfx1250RaceTest, tensor_iterations_newest_single_race) {
  auto *src = allocWithData<int>(kTileElements), *dst = alloc<int>(kTileElements);
  auto multi = describe({.global = src});
  multi.group1[0] |= 1u << 19; // Enable the iteration descriptor.
  tensor_iteration_count_kernel<true, 1, 4 * kTileBytes>
      <<<1, kWave, 5 * kTileBytes>>>(
          multi, describe({.global = src, .lds_offset = 4 * kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

TEST_F(Gfx1250RaceTest, tensor_iterations_newest_first_range_race) {
  auto *src = allocWithData<int>(kTileElements), *dst = alloc<int>(kTileElements);
  auto multi = describe({.global = src});
  multi.group1[0] |= 1u << 19; // Enable the iteration descriptor.
  tensor_iteration_count_kernel<false, 1, 0>
      <<<1, kWave, 5 * kTileBytes>>>(
          multi, describe({.global = src, .lds_offset = 4 * kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

TEST_F(Gfx1250RaceTest, tensor_iterations_do_not_add_counter_tokens_race) {
  auto *src = allocWithData<int>(kTileElements), *dst = alloc<int>(kTileElements);
  auto multi = describe({.global = src});
  multi.group1[0] |= 1u << 19; // Enable the iteration descriptor.
  tensor_iteration_count_kernel<false, 2, 4 * kTileBytes>
      <<<1, kWave, 5 * kTileBytes>>>(
          multi, describe({.global = src, .lds_offset = 4 * kTileBytes}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectRace(kTensorLoadThenReadRace);
}

// --- LDS addresses past the workgroup's allocation ---

// A 2048-byte allocation, one LDS block, with the tile's second half past it.
// Out-of-range LDS addresses do not abort descriptor processing (CDNA5 ISA
// 10.11.6); the in-range half transfers normally. Hardware may report MEMVIOL
// afterward when LDS_CONFIG.ADDR_OUT_OF_RANGE_REPORTING is enabled. These
// emulator tests do not exercise that reporting.
constexpr int kLdsAllocation = 2048;
constexpr uint32_t kStraddlingOffset = kLdsAllocation - kTileBytes / 2;

__global__ void tensor_load_past_lds_allocation_kernel(TensorDesc desc, int *dst) {
  tensorLoad(desc);
  waitTensorcnt<0>();
  if (lane() < 16)
    dst[lane()] = ldsReadAndWait(kStraddlingOffset + lane() * 4);
}

TEST_F(Gfx1250RaceTest, tensor_load_past_lds_allocation) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = allocFilled(kTileElements / 2, kUnwrittenValue);
  tensor_load_past_lds_allocation_kernel<<<1, kWave, kLdsAllocation>>>(
      describe({.global = src, .lds_offset = kStraddlingOffset}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0, kTileElements / 2));
}

__global__ void tensor_store_past_lds_allocation_kernel(TensorDesc desc) {
  if (lane() < 16)
    ldsWrite(kStraddlingOffset + lane() * 4, 100 + lane());
  waitDscnt();
  tensorStore(desc);
  waitTensorcnt<0>();
}

TEST_F(Gfx1250RaceTest, tensor_store_past_lds_allocation) {
  auto *dst = alloc<int>(kTileElements);
  tensor_store_past_lds_allocation_kernel<<<1, kWave, kLdsAllocation>>>(
      describe({.global = dst, .lds_offset = kStraddlingOffset}));
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  // The guide does not define the global values written from out-of-range LDS.
  std::vector<int> stored(kTileElements);
  RACE_HIP_CHECK(hipMemcpy(stored.data(), dst, kTileBytes, hipMemcpyDeviceToHost));
  stored.resize(kTileElements / 2);
  EXPECT_EQ(stored, range(100, kTileElements / 2));
}

// --- Compiler-generated tensor load ---

// The builtin issues the load; the compiler emits no s_wait_tensorcnt for it,
// so the kernel adds one, and schedules the LDS accesses and their waits.
__global__ void tensor_load_builtin_kernel(TensorDesc desc, int *dst) {
  __shared__ int tile[kTileElements];
  tile[threadIdx.x] = kUnwrittenValue;
  __syncthreads();
  const v4i zero4 = {0, 0, 0, 0};
  const v8i zero8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(desc.group0, desc.group1, zero4, zero4, zero8, 0);
  waitTensorcnt<0>();
  dst[threadIdx.x] = tile[threadIdx.x];
}

TEST_F(Gfx1250RaceTest, tensor_load_builtin) {
  auto *src = allocWithData<int>(kTileElements);
  auto *dst = alloc<int>(kWave);
  // `tile` is the kernel's only LDS allocation, so it starts at offset 0.
  tensor_load_builtin_kernel<<<1, kWave>>>(describe({.global = src}), dst);
  RACE_HIP_CHECK(hipGetLastError());
  sync();
  ExpectNoRace();
  ExpectValues(dst, range(0));
}

} // namespace

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <utility>

#include "cudepthfusion/error.hpp"

namespace cudepthfusion::cuda {

inline void check(cudaError_t status, const char* call, const char* file, int line) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << call << " failed at " << file << ":" << line << ": " << cudaGetErrorName(status)
            << " (" << cudaGetErrorString(status) << ")";
    throw CudaError(message.str());
  }
}

#define CUDEPTHFUSION_CUDA_CHECK(call) \
  ::cudepthfusion::cuda::check((call), #call, __FILE__, __LINE__)

// Owns one stream. Destructors never throw; a failure there can only be reported by the
// next checked call.
class Stream {
 public:
  Stream() { CUDEPTHFUSION_CUDA_CHECK(cudaStreamCreate(&stream_)); }
  ~Stream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
      }
      stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
  }

  cudaStream_t get() const { return stream_; }
  void synchronize() const { CUDEPTHFUSION_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  cudaStream_t stream_ = nullptr;
};

// Owns one timing event. Created once with the pipeline, recorded every frame.
class Event {
 public:
  Event() { CUDEPTHFUSION_CUDA_CHECK(cudaEventCreate(&event_)); }
  ~Event() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) {
        cudaEventDestroy(event_);
      }
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  void record(cudaStream_t stream) { CUDEPTHFUSION_CUDA_CHECK(cudaEventRecord(event_, stream)); }
  void synchronize() const { CUDEPTHFUSION_CUDA_CHECK(cudaEventSynchronize(event_)); }
  // Milliseconds from `start` to this event; both must have completed.
  double since(const Event& start) const {
    float elapsed = 0.0f;
    CUDEPTHFUSION_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.event_, event_));
    return static_cast<double>(elapsed);
  }

 private:
  cudaEvent_t event_ = nullptr;
};

// Device memory that only ever grows, so a steady frame loop allocates nothing (spec 7.6).
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        capacity_(std::exchange(other.capacity_, 0)),
        size_(std::exchange(other.size_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      capacity_ = std::exchange(other.capacity_, 0);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  void resize(std::size_t count) {
    if (count > capacity_) {
      release();
      void* raw = nullptr;
      CUDEPTHFUSION_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
      data_ = static_cast<T*>(raw);
      capacity_ = count;
    }
    size_ = count;
  }

  void release() {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
    capacity_ = 0;
    size_ = 0;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t capacity_bytes() const { return capacity_ * sizeof(T); }

  void upload(const T* host, std::size_t count, cudaStream_t stream) {
    CUDEPTHFUSION_CUDA_CHECK(
        cudaMemcpyAsync(data_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  }
  void download(T* host, std::size_t count, cudaStream_t stream) const {
    CUDEPTHFUSION_CUDA_CHECK(
        cudaMemcpyAsync(host, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
  void copy_from(const DeviceBuffer& other, std::size_t count, cudaStream_t stream) {
    CUDEPTHFUSION_CUDA_CHECK(
        cudaMemcpyAsync(data_, other.data_, count * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }
  void fill_bytes(int value, std::size_t count, cudaStream_t stream) {
    CUDEPTHFUSION_CUDA_CHECK(cudaMemsetAsync(data_, value, count * sizeof(T), stream));
  }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
};

// Page-locked host memory that only ever grows, used as the staging area for copies. Copies
// from pageable memory go through the driver's own bounce buffers and do not overlap; staging
// through one pinned block lets every download run at full bus speed back to back.
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { release(); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void reserve(std::size_t bytes) {
    if (bytes > capacity_) {
      release();
      void* raw = nullptr;
      CUDEPTHFUSION_CUDA_CHECK(cudaHostAlloc(&raw, bytes, cudaHostAllocDefault));
      data_ = static_cast<unsigned char*>(raw);
      capacity_ = bytes;
    }
  }

  void release() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
    }
    capacity_ = 0;
  }

  unsigned char* data() { return data_; }

 private:
  unsigned char* data_ = nullptr;
  std::size_t capacity_ = 0;
};

}  // namespace cudepthfusion::cuda

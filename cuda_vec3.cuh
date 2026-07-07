#ifndef CUDA_VEC3_CUH
#define CUDA_VEC3_CUH

#include <cuda_runtime.h>
#include <cmath>

struct CudaVec3 {
    double x, y, z;

    __host__ __device__ CudaVec3() : x(0), y(0), z(0) {}
    __host__ __device__ CudaVec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ CudaVec3 operator+(const CudaVec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    __host__ __device__ CudaVec3 operator-(const CudaVec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    __host__ __device__ CudaVec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    __host__ __device__ CudaVec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    __host__ __device__ CudaVec3 operator-() const { return {-x, -y, -z}; }
    __host__ __device__ CudaVec3& operator+=(const CudaVec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    __host__ __device__ CudaVec3& operator*=(double s) { x*=s; y*=s; z*=s; return *this; }

    __host__ __device__ double dot(const CudaVec3& v) const { return x*v.x + y*v.y + z*v.z; }
    __host__ __device__ CudaVec3 cross(const CudaVec3& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    __host__ __device__ double length() const { return sqrt(x*x + y*y + z*z); }
    __host__ __device__ double length2() const { return x*x + y*y + z*z; }
    __host__ __device__ CudaVec3 normalized() const {
        double l = length();
        return l > 0 ? *this / l : *this;
    }
    __host__ __device__ CudaVec3 mul(const CudaVec3& v) const { return {x*v.x, y*v.y, z*v.z}; }
    __host__ __device__ CudaVec3 rcp() const { return {1.0/x, 1.0/y, 1.0/z}; }
};

__host__ __device__ inline CudaVec3 operator*(double s, const CudaVec3& v) { return v * s; }

using CudaColor = CudaVec3;

#endif // CUDA_VEC3_CUH

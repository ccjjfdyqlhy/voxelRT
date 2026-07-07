#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

struct Vec3 {
    double x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s) const { return {x/s, y/s, z/s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    Vec3& operator*=(double s) { x*=s; y*=s; z*=s; return *this; }

    double dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    Vec3 cross(const Vec3& v) const {
        return {y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x};
    }
    double length() const { return std::sqrt(x*x + y*y + z*z); }
    double length2() const { return x*x + y*y + z*z; }
    Vec3 normalized() const { double l = length(); return l>0 ? *this/l : *this; }

    // 逐分量乘法
    Vec3 mul(const Vec3& v) const { return {x*v.x, y*v.y, z*v.z}; }
    // 逐分量倒数
    Vec3 rcp() const { return {1.0/x, 1.0/y, 1.0/z}; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v*s; }

// 颜色 = Vec3 (RGB in [0,1])
using Color = Vec3;

#endif // VEC3_H

#ifndef RAY_H
#define RAY_H

#include "vec3.h"

struct Ray {
    Vec3 origin;
    Vec3 dir;       // 单位方向向量

    Ray() = default;
    Ray(const Vec3& o, const Vec3& d) : origin(o), dir(d.normalized()) {}

    Vec3 at(double t) const { return origin + dir * t; }
};

#endif // RAY_H

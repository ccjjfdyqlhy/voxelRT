#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include "ray.h"
#include <cmath>

class Camera {
public:
    Camera(Vec3 pos = Vec3(0, 0, 0), double yaw = 0, double pitch = 0,
           double fov = 60, double aspect = 16.0/9.0)
        : position_(pos), yaw_(yaw), pitch_(pitch), fov_(fov), aspect_(aspect)
    {
        rebuild();
    }

    void setPosition(const Vec3& pos) { position_ = pos; rebuild(); }
    void setYaw(double y) { yaw_ = y; rebuild(); }
    void setPitch(double p) { pitch_ = std::max(-89.0, std::min(89.0, p)); rebuild(); }
    void setFov(double f) { fov_ = f; rebuild(); }
    void setAspect(double a) { aspect_ = a; rebuild(); }

    void move(const Vec3& delta) { position_ = position_ + delta; rebuild(); }

    void rotate(double dyaw, double dpitch) {
        yaw_ += dyaw;
        pitch_ += dpitch;
        pitch_ = std::max(-89.0, std::min(89.0, pitch_));
        rebuild();
    }

    Vec3 forward() const { return forward_; }
    Vec3 right() const { return right_; }
    Vec3 up() const { return up_; }
    Vec3 position() const { return position_; }

    Ray getRay(double u, double v) const {
        Vec3 dir = lowerLeft_ + horizontal_ * u + vertical_ * v - position_;
        return Ray(position_, dir);
    }

    double yaw() const { return yaw_; }
    double pitch() const { return pitch_; }

private:
    void rebuild() {
        // 从 yaw/pitch 计算 forward
        double cy = std::cos(yaw_ * M_PI / 180.0);
        double sy = std::sin(yaw_ * M_PI / 180.0);
        double cp = std::cos(pitch_ * M_PI / 180.0);
        double sp = std::sin(pitch_ * M_PI / 180.0);
        forward_ = Vec3(sy * cp, sp, cy * cp).normalized();

        Vec3 worldUp(0, 1, 0);
        right_ = worldUp.cross(forward_).normalized();
        up_ = forward_.cross(right_).normalized();

        double theta = fov_ * M_PI / 180.0;
        double h = std::tan(theta / 2.0);
        double vp_h = 2.0 * h;
        double vp_w = aspect_ * vp_h;

        w_ = forward_;  // view direction (into screen)
        horizontal_ = right_ * vp_w;
        vertical_ = up_ * vp_h;
        lowerLeft_ = position_ + w_ - horizontal_ / 2.0 - vertical_ / 2.0;
    }

    Vec3 position_;
    double yaw_, pitch_;
    double fov_, aspect_;
    Vec3 forward_, right_, up_;
    Vec3 w_;    // forward (view plane direction)
    Vec3 horizontal_, vertical_, lowerLeft_;
};

#endif // CAMERA_H

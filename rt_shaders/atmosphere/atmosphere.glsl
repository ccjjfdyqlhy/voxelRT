// rt_shaders - Precomputed Atmospheric Scattering (based on Derivative Shaders)
// Original: Bruneton et al. precomputed scattering model.
// Adapted to generic uniform interface.
// License: MIT

#ifndef RT_ATMOSPHERE_GLSL
#define RT_ATMOSPHERE_GLSL

#include "../core/math.glsl"

// --- Phase functions ---

float RayleighPhase(in float cosTheta) {
    const float c = 3.0 / 16.0 * rPI;
    return cosTheta * cosTheta * c + c;
}

float HenyeyGreensteinPhase(in float cosTheta, in const float g) {
    const float gg = g * g;
    float phase = 1.0 + gg - 2.0 * g * cosTheta;
    return oneMinus(gg) / (4.0 * PI * phase * sqrt(phase));
}

float CornetteShanksPhase(in float cosTheta, in const float g) {
    const float gg = g * g;
    float a = oneMinus(gg) * rcp(2.0 + gg) * 3.0 * rPI;
    float b = (1.0 + sqr(cosTheta)) * pow((1.0 + gg - 2.0 * g * cosTheta), -1.5);
    return a * b * 0.125;
}

float MiePhaseClouds(in float cosTheta, in const vec3 g, in const vec3 w) {
    const vec3 gg = g * g;
    vec3 a = (0.75 * oneMinus(gg)) * rcp(2.0 + gg);
    vec3 b = (1.0 + sqr(cosTheta)) * pow(1.0 + gg - 2.0 * g * cosTheta, vec3(-1.5));
    return dot(a * b, w) / (w.x + w.y + w.z);
}

vec3 DoNightEye(in vec3 color) {
    float luminance = GetLuminance(color);
    float rodFactor = exp2(-luminance * 6e2);
    return mix(color, luminance * vec3(0.72, 0.95, 1.2), rodFactor);
}

float fastAcos(in float x) {
    float a = abs(x);
    float r = 1.570796 - 0.175394 * a;
    r *= sqrt(1.0 - a);
    return x < 0.0 ? PI - r : r;
}

// --- Sky projection ---

vec2 ProjectSky(in vec3 direction) {
    vec2 coord = vec2(atan(-direction.x, -direction.z) * rTAU + 0.5, fastAcos(direction.y) * rPI);
    coord.x = coord.x * oneMinus(4.0 / 255.0) + 2.0 / 255.0;
    return saturate(coord * vec2(255.0, 256.0) * uScreenPixelSize);
}

vec3 UnprojectSky(in vec2 coord) {
    coord.x *= 256.0 / 255.0;
    coord.x = fract((coord.x - 2.0 / 255.0) / oneMinus(4.0 / 255.0));
    coord *= vec2(TAU, PI);
    return vec3(sincos(coord.x) * sin(coord.y), cos(coord.y)).xzy;
}

vec2 RaySphereIntersection(in vec3 pos, in vec3 dir, in float rad) {
    float PdotD = dot(pos, dir);
    float delta = sqr(PdotD) + sqr(rad) - dotSelf(pos);
    if (delta < 0.0) return vec2(-1.0);
    delta = sqrt(delta);
    return vec2(-delta, delta) - PdotD;
}

// --- Atmosphere model ---

const float planetRadius = 6371e3;
const float sun_angular_radius = 0.012;
const float mie_phase_g = 0.77;

vec3 lightningColor = vec3(0.45, 0.43, 1.0) * 0.03;

#ifdef AURORA
    float auroraAmount = smoothstep(0.0, 0.2, -uSunDir.y) * AURORA_STRENGTH;
#endif

#define TRANSMITTANCE_TEXTURE_WIDTH     256.0
#define TRANSMITTANCE_TEXTURE_HEIGHT    64.0
#define SCATTERING_TEXTURE_R_SIZE       32.0
#define SCATTERING_TEXTURE_MU_SIZE      128.0
#define SCATTERING_TEXTURE_MU_S_SIZE    32.0
#define SCATTERING_TEXTURE_NU_SIZE      8.0
#define IRRADIANCE_TEXTURE_WIDTH        64.0
#define IRRADIANCE_TEXTURE_HEIGHT       16.0

struct AtmosphereParameters {
    vec3 solar_irradiance;
    vec3 rayleigh_scattering;
    vec3 mie_scattering;
    vec3 ground_albedo;
};

AtmosphereParameters atmosphereModel = AtmosphereParameters(
    vec3(1.474000, 1.850400, 1.911980),
    vec3(0.005802, 0.013558, 0.033100),
    vec3(0.003996, 0.003996, 0.003996),
    vec3(0.1)
);

#define ATMOSPHERE_BOTTOM_ALTITUDE  1000.0
#define ATMOSPHERE_TOP_ALTITUDE     100000.0

const float atmosphere_bottom_radius = planetRadius - ATMOSPHERE_BOTTOM_ALTITUDE;
const float atmosphere_top_radius = planetRadius + ATMOSPHERE_TOP_ALTITUDE;
const float atmosphere_bottom_radius_sq = atmosphere_bottom_radius * atmosphere_bottom_radius;
const float atmosphere_top_radius_sq = atmosphere_top_radius * atmosphere_top_radius;
const float mu_s_min = -0.2;

float ClampCosine(float mu) { return clamp(mu, -1.0, 1.0); }
float ClampRadius(float r)  { return clamp(r, atmosphere_bottom_radius, atmosphere_top_radius); }
float SafeSqrt(float a)     { return sqrt(max0(a)); }

float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + atmosphere_top_radius_sq;
    return max0(-r * mu + SafeSqrt(discriminant));
}

float DistanceToBottomAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + atmosphere_bottom_radius_sq;
    return max0(-r * mu - SafeSqrt(discriminant));
}

bool RayIntersectsGround(float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + atmosphere_bottom_radius_sq >= 0.0;
}

float GetTextureCoordFromUnitRange(float x, float texture_size) {
    return 0.5 / texture_size + x * oneMinus(1.0 / texture_size);
}

vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
    const float H = sqrt(atmosphere_top_radius_sq - atmosphere_bottom_radius_sq);
    float rho = SafeSqrt(r * r - atmosphere_bottom_radius_sq);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = atmosphere_top_radius - r;
    float d_max = rho + H;
    return vec2(GetTextureCoordFromUnitRange((d - d_min) / (d_max - d_min), TRANSMITTANCE_TEXTURE_WIDTH),
                GetTextureCoordFromUnitRange(rho / H, TRANSMITTANCE_TEXTURE_HEIGHT));
}

vec3 GetTransmittanceToTopAtmosphereBoundary(float r, float mu) {
    vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
    uv = clamp(uv, vec2(0.5 / 256.0, 0.5 / 64.0), vec2(255.5 / 256.0, 63.5 / 64.0));
    return vec3(texture(uLut3D, vec3(uv * vec2(1.0, 0.5), 32.5 / 33.0)));
}

vec3 GetTransmittance(float r, float mu, float d, bool ray_r_mu_intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    if (ray_r_mu_intersects_ground) {
        return min(GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
                   GetTransmittanceToTopAtmosphereBoundary(r, -mu), vec3(1.0));
    } else {
        return min(GetTransmittanceToTopAtmosphereBoundary(r, mu) /
                   GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d), vec3(1.0));
    }
}

vec3 GetTransmittance(vec3 view_ray) {
    vec3 camera = vec3(0.0, planetRadius + uEyeAltitude, 0.0);
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top = -rmu - sqrt(rmu * rmu - r * r + atmosphere_top_radius_sq);
    if (distance_to_top > 0.0) {
        camera += view_ray * distance_to_top;
        r = atmosphere_top_radius;
        rmu += distance_to_top;
    } else if (r > atmosphere_top_radius) {
        return vec3(1.0);
    }
    float mu = rmu / r;
    return GetTransmittanceToTopAtmosphereBoundary(r, mu);
}

vec3 GetTransmittanceToSun(float r, float mu_s) {
    float sin_theta_h = atmosphere_bottom_radius / r;
    float cos_theta_h = -sqrt(max0(1.0 - sin_theta_h * sin_theta_h));
    return GetTransmittanceToTopAtmosphereBoundary(r, mu_s) *
           smoothstep(-sin_theta_h * sun_angular_radius,
                       sin_theta_h * sun_angular_radius,
                       mu_s - cos_theta_h);
}

vec4 GetScatteringTextureUvwzFromRMuMuSNu(float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground) {
    float H = sqrt(atmosphere_top_radius_sq - atmosphere_bottom_radius_sq);
    float rho = SafeSqrt(r * r - atmosphere_bottom_radius_sq);
    float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + atmosphere_bottom_radius_sq;
    float u_mu;
    if (ray_r_mu_intersects_ground) {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - atmosphere_bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE * 0.5);
    } else {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = atmosphere_top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange((d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE * 0.5);
    }
    float d = DistanceToTopAtmosphereBoundary(atmosphere_bottom_radius, mu_s);
    float d_min = atmosphere_top_radius - atmosphere_bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float D = DistanceToTopAtmosphereBoundary(atmosphere_bottom_radius, mu_s_min);
    float A = (D - d_min) / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(max0(1.0 - a / A) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);
    float u_nu = nu * 0.5 + 0.5;
    return vec4(u_nu, u_mu_s, u_mu, u_r);
}

vec3 GetExtrapolatedSingleMieScattering(AtmosphereParameters atmosphere, vec4 scattering) {
    if (scattering.r <= 0.0) return vec3(0.0);
    return scattering.rgb * scattering.a / scattering.r *
           (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
           (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}

vec3 GetCombinedScattering(AtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu,
                           bool ray_r_mu_intersects_ground, out vec3 single_mie_scattering) {
    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * (SCATTERING_TEXTURE_NU_SIZE - 1.0);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;
    vec3 uvw0 = vec3((tex_x + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE, uvwz.z, uvwz.w);
    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / SCATTERING_TEXTURE_NU_SIZE, uvwz.z, uvwz.w);
    vec4 combined_scattering = texture(uLut3D, uvw0) * oneMinus(lerp) + texture(uLut3D, uvw1) * lerp;
    vec3 scattering = vec3(combined_scattering);
    single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
    return scattering;
}

vec3 GetIrradiance(float r, float mu_s) {
    float x_r = (r - atmosphere_bottom_radius) / (atmosphere_top_radius - atmosphere_bottom_radius);
    float x_mu_s = mu_s * 0.5 + 0.5;
    vec2 uv = vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
                   GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
    uv = clamp(uv, vec2(0.5 / 64.0, 0.5 / 16.0), vec2(63.5 / 64.0, 15.5 / 16.0));
    return vec3(texture(uLut3D, vec3(uv * vec2(0.25, 0.125) + vec2(0.0, 0.5), 32.5 / 33.0)));
}

vec3 GetSkyRadiance(AtmosphereParameters atmosphere, vec3 view_ray, vec3 sun_direction, out vec3 transmittance) {
    vec3 camera = vec3(0.0, planetRadius + uEyeAltitude, 0.0);
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top = -rmu - sqrt(rmu * rmu - r * r + atmosphere_top_radius_sq);
    if (distance_to_top > 0.0) {
        camera += view_ray * distance_to_top;
        r = atmosphere_top_radius;
        rmu += distance_to_top;
    } else if (r > atmosphere_top_radius) {
        transmittance = vec3(1.0);
        return vec3(0.0);
    }
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);
    transmittance = ray_r_mu_intersects_ground ? vec3(0.0) : GetTransmittanceToTopAtmosphereBoundary(r, mu);

    vec3 sun_single_mie_scattering, moon_single_mie_scattering;
    vec3 sun_scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, sun_single_mie_scattering);
    vec3 moon_scattering = GetCombinedScattering(atmosphere, r, mu, -mu_s, -nu, ray_r_mu_intersects_ground, moon_single_mie_scattering);

    vec3 rayleigh = sun_scattering * RayleighPhase(nu) + moon_scattering * RayleighPhase(-nu) * NIGHT_BRIGHTNESS;
    vec3 mie = sun_single_mie_scattering * HenyeyGreensteinPhase(nu, mie_phase_g)
             + moon_single_mie_scattering * HenyeyGreensteinPhase(-nu, mie_phase_g) * NIGHT_BRIGHTNESS;
    rayleigh = mix(rayleigh, GetLuminance(rayleigh) * vec3(1.026186824, 0.9881671071, 1.015787125), uWetness * 0.7);
    return (rayleigh + mie) * oneMinus(uWetness * 0.6);
}

vec3 GetSkyRadianceToPoint(AtmosphereParameters atmosphere, vec3 point, vec3 sun_direction, out vec3 transmittance) {
    vec3 camera = vec3(0.0, planetRadius + uEyeAltitude, 0.0);
    vec3 view_ray = normalize(point);
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top = -rmu - sqrt(rmu * rmu - r * r + atmosphere_top_radius_sq);
    if (distance_to_top > 0.0) {
        camera += view_ray * distance_to_top;
        r = atmosphere_top_radius;
        rmu += distance_to_top;
    }
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    float d = length(point);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);
    transmittance = GetTransmittance(r, mu, d, ray_r_mu_intersects_ground);
    vec3 sun_single_mie_scattering, moon_single_mie_scattering;
    vec3 sun_scattering = GetCombinedScattering(atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground, sun_single_mie_scattering);
    vec3 moon_scattering = GetCombinedScattering(atmosphere, r, mu, -mu_s, -nu, ray_r_mu_intersects_ground, moon_single_mie_scattering);

    vec3 r_d = vec3(1.0);
    float mu_d = 1.0;
    if (distance_to_top <= 0.0) {
        r_d = point + camera;
        float r_d_len = length(r_d);
        mu_d = dot(r_d, sun_direction) / r_d_len;
        r_d_len = ClampRadius(r_d_len);
        transmittance = GetTransmittanceToTopAtmosphereBoundary(r, mu) / GetTransmittanceToTopAtmosphereBoundary(r_d_len, mu_d);
    }

    vec3 rayleigh = sun_scattering * RayleighPhase(nu) + moon_scattering * RayleighPhase(-nu) * NIGHT_BRIGHTNESS;
    vec3 mie = sun_single_mie_scattering * HenyeyGreensteinPhase(nu, mie_phase_g)
             + moon_single_mie_scattering * HenyeyGreensteinPhase(-nu, mie_phase_g) * NIGHT_BRIGHTNESS;
    return (rayleigh + mie) * oneMinus(uWetness * 0.6);
}

#endif // RT_ATMOSPHERE_GLSL

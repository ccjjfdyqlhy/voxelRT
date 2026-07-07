// rt_shaders - Universal configuration
// All settings are generic — no Minecraft concepts.
// License: MIT

#ifndef RT_CONFIG_GLSL
#define RT_CONFIG_GLSL

// ============ Atmospherics ============

#define TEMPORAL_UPSCALING 2        // [2 3 4]
#define MAX_BLENDED_FRAMES 40.0

#define PLANAR_CLOUDS
#define VOLUMETRIC_CLOUDS
#define CIRRUS_CLOUDS 1             // [0 1 2]
#define CIRROCUMULUS_CLOUDS
#define CLOUDS_SPEED 1.0
#define minTransmittance 0.05

// #define CLOUDS_SHADOW
// #define PC_SHADOW
// #define VC_SHADOW
// #define CLOUDS_WEATHER

// #define AURORA
#define AURORA_STRENGTH 0.7

#define STARS_INTENSITY 0.1
#define STARS_COVERAGE  0.15

#define LAND_ATMOSPHERIC_SCATTERING
// #define VOLUMETRIC_FOG
// #define VOLUMETRIC_LIGHT
// #define UW_VOLUMETRIC_LIGHT

#define FOG_TYPE 1                  // [0 1 2 3]

#define SUNLIGHT_INTENSITY 1.0
#define SKYLIGHT_INTENSITY 1.0
#define NIGHT_BRIGHTNESS 0.0005

// ============ Shadows ============

#define SHADOW_MAP_BIAS 0.9
// #define COLORED_SHADOWS
// #define SCREEN_SPACE_SHADOWS
// #define SHADOW_BACKFACE_CULLING

const int shadowMapResolution = 2048;
const float shadowDistance = 192.0;

#define PCF_SAMPLES 16             // [4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 48 64]

// ============ GI & AO ============

// #define GI_ENABLED
#define GI_SAMPLES 16
#define GI_RADIUS 30
#define GI_BRIGHTNESS 1.0

// #define SSAO_ENABLED
#define SSAO_SAMPLES 6
#define SSAO_STRENGTH 1.0

// ============ Surface ============

#define SPECULAR_HIGHLIGHT_BRIGHTNESS 0.6
#define SUBSERFACE_SCATTERING_MODE 0
#define SUBSERFACE_SCATTERING_STRENTGH 1.0
// #define ROUGH_REFLECTIONS
#define ROUGH_REFLECTIONS_THRESHOLD 0.005
// #define REFLECTION_FILTER

// #define PARALLAX
#define PARALLAX_SAMPLES 60
#define PARALLAX_DEPTH 0.2
#define PARALLAX_SHADOW
#define PARALLAX_REFINEMENT
#define PARALLAX_REFINEMENT_STEPS 8

// ============ Water ============

#define WATER_REFRACT_IOR 1.33
// #define WATER_CAUSTICS
#define WATER_FOG_DENSITY 1.0
#define WATER_WAVE_HEIGHT 1.0
#define WATER_WAVE_SPEED 1.0
// #define WATER_PARALLAX

// ============ Post Processing ============

// #define DOF_ENABLED
#define FOCUSING_SPEED 6.0
// #define MOTION_BLUR
#define MOTION_BLUR_SAMPLES 6
#define MOTION_BLUR_STRENGTH 0.5

// #define BLOOM_ENABLED
#define BLUR_SAMPLES 1
#define BLOOM_AMOUNT 1.0
// #define BLOOMY_FOG

// #define TAA_ENABLED
#define TAA_SHARPNESS 0.7

// #define AUTO_EXPOSURE
#define AUTO_EXPOSURE_LOD 6
#define EXPOSURE_SPEED 1.0
#define AUTO_EXPOSURE_BIAS 0.0
#define MANUAL_EXPOSURE_VALUE 12.0

// #define CAS_ENABLED
#define CAS_STRENGTH 0.3

// ============ Debug ============

#define DEBUG_NORMAL 0              // [0 1 2]

#endif // RT_CONFIG_GLSL

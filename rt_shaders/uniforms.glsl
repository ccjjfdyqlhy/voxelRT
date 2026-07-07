// rt_shaders - Universal uniform declarations
// Replace these with your engine's actual uniforms.
// License: MIT

#ifndef RT_UNIFORMS_GLSL
#define RT_UNIFORMS_GLSL

// --- Textures ---
uniform sampler2D uNoiseTex;

uniform sampler2D uTexAlbedo;    // gbuffer albedo
uniform sampler2D uTexNormal;    // gbuffer normal
uniform sampler2D uTexDepth;     // depth buffer
uniform sampler2D uTexLightmap;  // lightmap / sky-occlusion
uniform sampler2D uTexMaterial;  // material ID / roughness / metal
uniform sampler2D uTexSpecular;  // specular color
uniform sampler2D uTexEmission;  // emission
uniform sampler2D uTexPrevColor; // previous frame (for TAA / temporal)

uniform sampler2D uShadowMap;       // shadow depth
uniform sampler2D uShadowColor;     // shadow colored transparency
uniform sampler2D uShadowNormal;    // shadow normal buffer (RSM GI)
uniform sampler2D uShadowAlbedo;    // shadow albedo buffer (RSM GI)

uniform sampler3D uLut3D;          // 3D LUT for scattering / transmittance

// --- Matrices ---
uniform mat4 uViewMatrix;
uniform mat4 uViewMatrixInv;
uniform mat4 uProjMatrix;
uniform mat4 uProjMatrixInv;
uniform mat4 uPrevViewMatrix;
uniform mat4 uPrevProjMatrix;
uniform mat4 uShadowMatrixView;
uniform mat4 uShadowMatrixProj;
uniform mat4 uShadowMatrixProjInv;

// --- Camera ---
uniform vec3  uCameraPos;
uniform vec3  uPrevCameraPos;
uniform float uNear;
uniform float uFar;
uniform float uAspectRatio;
uniform float uFOV;

// --- Lighting ---
uniform vec3  uSunDir;        // normalized sun direction (points toward sun)
uniform vec3  uLightDir;      // normalized light direction (for shading)
uniform vec3  uSunIlluminance;
uniform vec3  uSkyIlluminance;
uniform vec3  uAmbientIlluminance;

// --- Environment ---
uniform float uEyeAltitude;   // height above sea level
uniform float uWetness;       // 0..1 rain wetness
uniform float uTime;          // seconds
uniform int   uFrameCounter;

// --- Screen ---
uniform float uScreenWidth;
uniform float uScreenHeight;
uniform vec2  uScreenSize;
uniform vec2  uScreenPixelSize;
uniform vec2  uTAAOffset;

#endif // RT_UNIFORMS_GLSL

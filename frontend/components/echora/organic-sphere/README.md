# Organic Sphere source

Reference: https://github.com/brunosimon/organic-sphere
Upstream commit inspected: `3a6c01e0e7ebb4aa8d73f015ec01a498242a730d`

`vertex.glsl`, `fragment.glsl`, and `perlin4d.glsl` are copied from Bruno Simon’s Organic Sphere at the user’s explicit request to use the original implementation. Original shader comments are retained; the Perlin implementation credits Stefan Gustavson. The reference also credits Drawfish’s Alien Ripple as visual inspiration.

`renderSphere.ts` retains the reference camera, 512 × 512 geometry, computed tangents and Perlin deformation. The original shader files are preserved; glslify includes are resolved at bundle time, excluding the unused 3D noise function. The vertex shader is extended at bundle time to pass displaced positions and normals to `pearl.glsl`.

Light-background adaptation: `pearl.glsl` uses a continuous opaque pearl surface, broad coral/blue lighting and a soft white highlight. Lighting follows the displaced normals so folds remain visible across the surface. Only the canvas outside the mesh is transparent. The former intensity-based transparency and bloom/composite passes are removed: those produced an empty white centre with harsh saturated outlines. The static fallback uses the same pale palette.

The sphere stays visible at the reference idle speed (.3 shader seconds per second), with gentler displacement (.12) and distortion (.42). Noise-gated microphone input is boosted with a .65 power curve, driving displacement up to .38, distortion up to .8, speed up to 1.7 and broader coral/blue coverage. The existing microphone envelope provides smooth attack/release without a second onset delay. Processing separately adds .12 to animation speed. Pixel ratio/frame rate limits, pause/reduced motion, disposal and static fallback remain. Transcription is unchanged.

import * as THREE from 'three';
import vertexSource from './vertex.glsl?raw';
import fragmentSource from './pearl.glsl?raw';
import perlin4d from './perlin4d.glsl?raw';

// Keep Bruno Simon's deformation; expose its normals to the pearl material.
// perlin3d is unused in the original vertex shader and does not need bundling.
const vertexShader = vertexSource
  .replace(
    "#pragma glslify: perlin4d = require('../partials/perlin4d.glsl')",
    perlin4d.replace(/#pragma glslify: export.*$/gm, ''),
  )
  .replace(
    "#pragma glslify: perlin3d = require('../partials/perlin3d.glsl')",
    '',
  )
  .replace('varying vec3 vColor;', 'varying vec3 vColor; varying vec3 vSurfaceNormal; varying vec3 vSurfacePosition;')
  .replace('computedNormal = normalize(computedNormal);', `
    computedNormal = normalize(computedNormal);
    vSurfaceNormal = computedNormal;
    vSurfacePosition = displacedPosition;
  `);

export function createOrganicSphere(renderer: THREE.WebGLRenderer) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(25, 1, 0.1, 15);
  camera.position.set(0, 0, 7);
  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader: fragmentSource,
    defines: { USE_TANGENT: '' },
    uniforms: {
      // Display-space palette: softly shaded pearl, copper and cornflower blue.
      uLightAColor: { value: new THREE.Color().setRGB(0.91, 0.43, 0.28) },
      uLightAPosition: {
        value: new THREE.Vector3().setFromSpherical(
          new THREE.Spherical(1, 0.615, 2.049),
        ),
      },
      uLightAIntensity: { value: 1.85 },
      uLightBColor: { value: new THREE.Color().setRGB(0.26, 0.48, 0.79) },
      uLightBPosition: {
        value: new THREE.Vector3().setFromSpherical(
          new THREE.Spherical(1, 2.561, -1.844),
        ),
      },
      uLightBIntensity: { value: 1.4 },
      uSubdivision: { value: new THREE.Vector2(512, 512) },
      uOffset: { value: new THREE.Vector3() },
      uDistortionFrequency: { value: 1.5 },
      uDistortionStrength: { value: 0.42 },
      uDisplacementFrequency: { value: 2.12 },
      uDisplacementStrength: { value: 0.12 },
      uFresnelOffset: { value: -1.609 },
      uFresnelMultiplier: { value: 3.587 },
      uFresnelPower: { value: 1.793 },
      uTime: { value: 0 },
      uVoiceEnergy: { value: 0 },
    },
  });
  const geometry = new THREE.SphereGeometry(1, 512, 512);
  geometry.computeTangents();
  scene.add(new THREE.Mesh(geometry, material));

  // Opaque pearl shading keeps volume; canvas transparency is only outside the mesh.
  return {
    resize(width: number, height: number) {
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    },
    update(dt: number, energy: number, processing: boolean, still = false) {
      if (still) return false;
      // VoiceMotion already gates noise and smooths attack/release. Boost normal
      // speech without another onset delay, while keeping loud input bounded.
      const voice = Math.pow(Math.min(1, Math.max(0, energy)), 0.65);
      const speed = 0.3 + voice * 1.4 + (processing ? 0.12 : 0);
      material.uniforms.uTime.value += dt * speed;
      // Follow the reference's offset direction, with bounded microphone input.
      material.uniforms.uOffset.value.x -= dt * speed * 0.12;
      material.uniforms.uDisplacementStrength.value = 0.12 + voice * 0.26;
      material.uniforms.uDistortionStrength.value = 0.42 + voice * 0.38;
      material.uniforms.uVoiceEnergy.value = voice;
      return true;
    },
    render() {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
    },
    dispose() {
      geometry.dispose();
      material.dispose();
    },
  };
}

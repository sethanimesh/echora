uniform vec3 uLightAColor;
uniform vec3 uLightAPosition;
uniform vec3 uLightBColor;
uniform vec3 uLightBPosition;
uniform float uVoiceEnergy;

varying vec3 vSurfaceNormal;
varying vec3 vSurfacePosition;

void main()
{
    vec3 normal = normalize(vSurfaceNormal);
    vec3 view = normalize(cameraPosition - vSurfacePosition);
    float facing = clamp(dot(normal, view), 0.0, 1.0);
    float rim = pow(1.0 - facing, 0.85);

    // Broad lighting follows the displaced normals through each fold, instead
    // of isolating a thin, saturated Fresnel outline around an empty centre.
    float warm = pow(max(dot(normal, normalize(uLightAPosition)), 0.0), 1.3);
    float cool = pow(max(dot(normal, normalize(uLightBPosition)), 0.0), 1.3);
    // Let the colour travel farther across the surface during speech.
    warm *= 0.22 + rim * 0.62 + uVoiceEnergy * 0.32;
    cool *= 0.22 + rim * 0.68 + uVoiceEnergy * 0.32;
    float total = max(1.0, (warm + cool) / 0.92);
    warm /= total;
    cool /= total;
    vec3 pearl = mix(vec3(0.87, 0.89, 0.93), vec3(0.98), pow(facing, 0.45));
    vec3 color = pearl * (1.0 - warm - cool) + uLightAColor * warm + uLightBColor * cool;

    vec3 softbox = normalize(vec3(-0.4, 0.8, 2.0));
    vec3 halfway = normalize(softbox + view);
    float highlight = pow(max(dot(normal, halfway), 0.0), 24.0) * 0.22;
    color = mix(color, vec3(1.0), highlight);
    gl_FragColor = vec4(color, 1.0);
}

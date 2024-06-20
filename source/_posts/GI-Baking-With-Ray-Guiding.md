---
title: GI Baking With Ray Guiding
date: 2024-06-14 23:08:32
index_img: /resource/gi_rayguiding/image/result.png
tags:
---

# Introduction

This project is part of HWRTL. HWRTL is a single file hardware ray tracing library containing GI baking, HLOD texture generation and DXR PVS generation.

Firstly, we pack the lightmaps of each mesh into a number of atlases and search for the best lightmap layout with the **smallest** total size. The packing algorithm uses a third-party library: **stb_pack**. Then, generate the **lightmap gbuffer** by GPU and start light map path tracing. For each texels in the light map gbuffer, we trace 512 rays from the world position recorded in the gbuffer and each tracing bounces 32 times. For each bounce, we perform a light sampling and material sampling and combine them by **multi-importance sampling** to reduce variance.

Furthermore, we have implemented part of the **ray-guiding algorithm**. First, we split the light map into **clusters** and each cluster shares the same luminance distribution, which can amplify the number of learning rays and reduce variance in the learned ray distribution. Then, map the ray sample direction of the first bounce into 2D, store the luminance at the corresponding position in the cluster and build the luminance distribution. After that, we build the luminance **CDF** in the compute shader by wave instruction. During the latter path tracing pass, we **adjust the sample direction and PDF** based on the ray guiding CDF.

We perform a simple **dilate pass** in order to solve the sampling bleed problem caused by bilinear sampling.

Unreal GPULM employs **Odin**(Open Image Denoise) to denoise the traced lightmap. We have implemented a custom denoising algorithm (**Non-local means filtering**)  to replace the Odin, since HWRTL design goal is a single file hardware raytracing library and we don't want to involve the thirdparty.

# Packing

The diffuse lightmap UV which is usually called 2U is different from the other texture UV coordinates. Multipile meshes may share one lightmap atlas, which means we should pack their GI baking results together. 

In HWRTL, the size of the mesh lightmap is specified by the user for simplicity. In practice, it's better to calculate the lightmap size based on the triangle size in the world and UV space.

We use the exhaunst algorithm to search for the result. First, search the maximum lightmap size among the scene meshes and use this size as the primary search size.

```cpp
std::vector<Vec2i> giMeshLightMapSize;
for (uint32_t index = 0; index < pGiBaker->m_giMeshes.size(); index++)
{
    SGIMesh& giMeshDesc = pGiBaker->m_giMeshes[index];
    giMeshLightMapSize.push_back(giMeshDesc.m_nLightMapSize);
    nAtlasSize.x = std::max(nAtlasSize.x, giMeshDesc.m_nLightMapSize.x + 2);
    nAtlasSize.y = std::max(nAtlasSize.y, giMeshDesc.m_nLightMapSize.y + 2);
}

int nextPow2 = NextPow2(nAtlasSize.x);
nextPow2 = std::max(nextPow2, NextPow2(nAtlasSize.y));

nAtlasSize = Vec2i(nextPow2, nextPow2);
```

Then, iterate the light map packing size from the maximum lightmap size to the user-specified maximum atlas size. 

For each iteration, try to pack them into a set of lightmap atlas using the open source library stb_rect_pack. If the total lightmap area is smaller than the minimum area computed in the previous iteration, use the current iteration lightmap layout results as the preferred layout.

<p align="center">
    <img src="/resource/gi_rayguiding/image/packing_vis.png" width="50%" height="50%">
</p>

Record the lightmap offset and scale of each mesh in the lightmap atlas texture.

```cpp
for (uint32_t index = 0; index < pGiBaker->m_giMeshes.size(); index++)
{
    SGIMesh& giMeshDesc = pGiBaker->m_giMeshes[index];
    giMeshDesc.m_nAtlasOffset = Vec2i(bestAtlasOffsets[index].x, bestAtlasOffsets[index].y);
    giMeshDesc.m_nAtlasIndex = bestAtlasOffsets[index].z;
    Vec2 scale = Vec2(giMeshDesc.m_nLightMapSize) / Vec2(bestAtlasSize);
    Vec2 bias = Vec2(giMeshDesc.m_nAtlasOffset) / Vec2(bestAtlasSize);
    Vec4 scaleAndBias = Vec4(scale.x, scale.y, bias.x, bias.y);
    giMeshDesc.m_lightMapScaleAndBias = scaleAndBias;
}
```
# GBuffer Generation

Generate the lightmap GBuffer for each lightmap atlas, which is GPU friendly for hardware raytracing. The pixel output destination is the texel position calculated from the lightmap UV and scale, which is located in atlas space rather than world space.

```cpp
float2 lightMapCoord = IN.m_lightmapuv * m_lightMapScaleAndBias.xy + m_lightMapScaleAndBias.zw;
vs2PS.m_position = float4((lightMapCoord - float2(0.5,0.5)) * float2(2.0,-2.0),0.0,1.0);
```
lightmap gbuffer world position and world normal:

<p align="center">
    <img src="/resource/gi_rayguiding/image/lm_gbuffer_world_norm.png" width="30%" height="30%">
</p>

<p align="center">
    <img src="/resource/gi_rayguiding/image/lm_gbuffer_world_pos.png" width="30%" height="30%">
</p>


# Baking

Our implementation references Unreal GPU Light Mass's implementation (**GPULM**). GPULM employ multi importance sampling algorithm.

## Multi Importance Sampling

Consider the problem of evaluating direct lighting integrals of the form

{% katex %} L_{o}(p,w_{o}) = \int_{S^{2}}f(p,w_{o},w_{i})L_{d}(p,w_{i})|cos\theta_{i}|\mathrm{d}w_{i}{% endkatex %}<br><br>

If we were to perform importance sampling to estimate this integral according to distributions based on either Ld or fr, one of these two will often perform poorly[<u>**[PBRT-V4]**</u>](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling). 

GPULM uses two sampling distributions to estimate the value of {% katex %}\int f(x)g(x)\mathrm{d}(x){% endkatex %} in order to solve this problem. The new Monte Carlo estimator given by MIS is:

{% katex %}\frac{1}{n_{f}}\Sigma_{i=1}^{n_{f}}\frac{f(X_{i})g(X_{i})w_{f}(X_{i})}{p_{f}(X_{i})} + \frac{1}{n_{g}}\Sigma_{j=1}^{n_{g}}\frac{f(Y_{j})g(Y_{j})w_{g}(Y_{j})}{p_{g}(Y_{j})}{% endkatex %}<br><br>

Considering that our lighting sampling count of the time is equal to our BRDF sampling count of the time, the final result is:

{% katex %}\frac{1}{n}\Sigma_{i=1}^{n}(\frac{f(X_{i})g(X_{i})w_{f}(X_{i})}{p_{f}(X_{i})} + \frac{f(Y_{i})g(Y_{i})w_{g}(Y_{i})}{p_{g}(Y_{i})}){% endkatex %}<br><br>

Given a sample X at a point where the value {% katex %}p_{f}(x){% endkatex %}is relatively low. Assuming that {% katex %}p_{f}(x){% endkatex %} is a good match for the shape of {% katex %}f(x){% endkatex %}, then the value of will also be relatively low. But suppose that {% katex %}g(x){% endkatex %} has a relatively high value. 

The standard importance sampling estimate {% katex %}\frac{f(x)g(x)}{p_{f}(x)}{% endkatex %}will have a very large value due to pf being small, and we will have high variance.

GPULM uses power heuristic weights with a power of two to reduce variance:
```cpp
float MISWeightRobust(float Pdf, float OtherPdf) 
{
	if (Pdf == OtherPdf)
	{
		return 0.5f;
	}

	if (OtherPdf < Pdf)
	{
		float x = OtherPdf / Pdf;
		return 1.0 / (1.0 + x * x);
	}
	else
	{
		float x = Pdf / OtherPdf;
		return 1.0 - 1.0 / (1.0 + x * x);
	}
}
```

## Light Map Path Tracing

Perform ray tracing for each texel in the lightmap GBuffer.

Obtain the world position and world normal from the lightmap buffer first.

```cpp
float3 worldPosition = rtWorldPosition[rayIndex].xyz;
float3 worldFaceNormal = rtWorldNormal[rayIndex].xyz;
```

Trace a ray from the world position and each ray bounce 32 time in our example.

For each bounce, we perform a light importance sampling and a material importance sampling.

```cpp
    for(int bounce = 0; bounce <= maxBounces; bounce++)
    {
        if(bIsCameraRay)
        {
            rtRaylod.m_worldPosition = worldPosition;
            rtRaylod.m_worldNormal = faceNormal;
            rtRaylod.m_vHiTt = 1.0f;
            rtRaylod.m_eFlag |= RT_PAYLOAD_FLAG_FRONT_FACE;
            rtRaylod.m_roughness = 1.0f;
            rtRaylod.m_baseColor = float3(1,1,1);
            rtRaylod.m_diffuseColor = float3(1,1,1);
            rtRaylod.m_specColor = float3(0,0,0);
        }
        else
        {
            rtRaylod = TraceLightRay(ray,bIsLastBounce,pathThroughput,radiance);
        }

        // ......
        // sampling
        
        // step1: Sample Light, Select a [LIGHT] randomly
        if(debugSample != 1)
        {
            if (vLightPickingCdfPreSum > 0)
            {
                // ......
                // light sampling
            }
        }

        // step2: Sample Material, Generate a [LIGHT DIRECTION] based on the material randomly
        if(debugSample != 2)
        {
            SMaterialSample materialSample = SampleMaterial(rtRaylod,randomSample);
        }
    }
```

### Light Importance Sampling

Generate the lighting picking cumulative distribution function first for inversion sampling method[<u>**[PBRT-V4]**</u>](https://pbr-book.org/4ed/Monte_Carlo_Integration/Sampling_Using_the_Inversion_Method).

we can draw a sample {% katex %} X_{i}  {% endkatex %} from a PDF p(x) with the following steps:

1.Integrate the PDF  to find the CDF {% katex %} P(x) = \int_{0}^{x}p(x^{'})\mathrm{d}x^{'}  {% endkatex %}. For discrete case, the result is: {% katex %} P(x) = \sum_{i=0}^{n}p(light_{i}){% endkatex %}:

```cpp
for(uint index = 0; index < m_nRtSceneLightCount; index++)
{
    vLightPickingCdfPreSum += EstimateLight(index,worldPosition,worldNormal);
    aLightPickingCdf[index] = vLightPickingCdfPreSum;
}
```
Estimate the light PDF for each light source in the scene:
```cpp
float GetLightFallof(float vSquaredDistance,int nLightIndex)
{
    float invLightAttenuation = 1.0 / rtSceneLights[nLightIndex].m_vAttenuation;
    float normalizedSquaredDistance = vSquaredDistance * invLightAttenuation * invLightAttenuation;
    return saturate(1.0 - normalizedSquaredDistance) * saturate(1.0 - normalizedSquaredDistance);
}

float EstimateDirectionalLight(uint nlightIndex)
{
    return Luminance(rtSceneLights[nlightIndex].m_color);
}

float EstimateSphereLight(uint nlightIndex,float3 worldPosition)
{
    float3 lightDirection = float3(rtSceneLights[nlightIndex].m_worldPosition - worldPosition);
    float squaredLightDistance = dot(lightDirection,lightDirection);
    float lightPower = Luminance(rtSceneLights[nlightIndex].m_color);

    return lightPower * GetLightFallof(squaredLightDistance,nlightIndex) / squaredLightDistance;
}

float EstimateLight(uint nlightIndex, float3 worldPosition,float3 worldNormal)
{
    switch(rtSceneLights[nlightIndex].m_eLightType)
    {
        case RT_LIGHT_TYPE_DIRECTIONAL: return EstimateDirectionalLight(nlightIndex);
        case RT_LIGHT_TYPE_SPHERE: return EstimateSphereLight(nlightIndex,worldPosition);
        default: return 0.0;
    }
}
```
2.Obtain a uniformly distributed random number {% katex %} \xi  {% endkatex %}.
3.Generate a sample {% katex %} \xi = P(X) {% endkatex %}by solving  for X; in other words, find {% katex %}X = P^{-1}(\xi){% endkatex %}.

```cpp
void SelectLight(float vRandom, int nLights, inout float aLightPickingCdf[RT_MAX_SCENE_LIGHT], out uint nSelectedIndex, out float vLightPickPdf)
{
    float preCdf = 0;
    for(nSelectedIndex = 0; nSelectedIndex < nLights;nSelectedIndex++)
    {
        if(vRandom < aLightPickingCdf[nSelectedIndex])
        {
            break;
        }
        preCdf = aLightPickingCdf[nSelectedIndex];
    }

    vLightPickPdf = aLightPickingCdf[nSelectedIndex] - preCdf;
}
```

Compute the direct lighting result of the sample.
{% katex %} L_{o}(p,w_{o}) = \int_{S^{2}}f(p,w_{o},w_{i})L_{d}(p,w_{i})Vis(p,w_{i})|cos\theta_{i}|\mathrm{d}w_{i}{% endkatex %}<br><br>

1.Sample the light based on the type of light sample selected before.
```cpp
SLightSample SampleLight(int nLightIndex,float2 vRandSample,float3 vWorldPos,float3 vWorldNormal)
{
    switch(rtSceneLights[nLightIndex].m_eLightType)
    {
        case RT_LIGHT_TYPE_DIRECTIONAL: return SampleDirectionalLight(nLightIndex,vRandSample,vWorldPos,vWorldNormal);
        case RT_LIGHT_TYPE_SPHERE: return SampleSphereLight(nLightIndex,vRandSample,vWorldPos,vWorldNormal);
        default: return (SLightSample)0;
    }
}
```
2.Compute the visibility term by tracing a shadow ray from the hit position to the light sample position.
```cpp
// trace a visibility ray
{
    SMaterialClosestHitPayload shadowRayPaylod = (SMaterialClosestHitPayload)0;
    
    RayDesc shadowRay;
    shadowRay.Origin = worldPosition;
    shadowRay.TMin = 0.0f;
    shadowRay.Direction = lightSample.m_direction;
    shadowRay.TMax = lightSample.m_distance;
    shadowRay.Origin += abs(worldPosition) * 0.001f * worldNormal;

    TraceRay(rtScene, RAY_FLAG_FORCE_OPAQUE, RAY_TRACING_MASK_OPAQUE, RT_SHADOW_SHADER_INDEX, 1,0, shadowRay, shadowRayPaylod);
    
    float sampleContribution = 0.0;
    if(shadowRayPaylod.m_vHiTt <= 0)
    {
        sampleContribution = 1.0;
    }
    lightSample.m_radianceOverPdf *= sampleContribution;
}
```

The ray bias calculation mathod is based on the [<u>**[Bakery LightMap Baker]**</u>](https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/).

```cpp
shadowRay.Origin += abs(worldPosition) * 0.001f * worldNormal;
```

3.Compute the BRDF term. GPULM uses simple **Lambertian** BRDF to calculate diffuse reflection and combines it with multi-importance sampling.
```cpp
if (any(lightSample.m_radianceOverPdf > 0))
{   
    SMaterialEval materialEval = EvalMaterial(lightSample.m_direction,rtRaylod);
    float3 lightContrib = pathThroughput * lightSample.m_radianceOverPdf * materialEval.m_weight * materialEval.m_pdf;

    lightContrib *= MISWeightRobust(lightSample.m_pdf,materialEval.m_pdf);
}
```

### Material Importance Sampling
1.Sample cosine-weighted hemisphere to compute wi and pdf:
```cpp
SMaterialSample SampleLambertMaterial(SMaterialClosestHitPayload payload, float4 randomSample)
{
    float3 worldNormal = payload.m_worldNormal;
    float4 sampleValue = CosineSampleHemisphere(randomSample.xy);

    SMaterialSample materialSample = (SMaterialSample)0;
    materialSample.m_direction = TangentToWorld(sampleValue.xyz,worldNormal);
    materialSample.m_pdf = sampleValue.w;
    return materialSample;
}
```
Assign the ray direction to the new generated ray direction that will be used in the next bounce.

```cpp
ray.Direction = normalize(materialSample.m_direction);
```

2.Calculate the incident radiance in the ray direction and trace a shadow ray to compute the visibility term by hardware raytracing.

```cpp
for(uint index = 0; index < m_nRtSceneLightCount; index++)
{
    SLightTraceResult lightTraceResult = TraceLight(ray,index);

    // ......

    RayDesc shadowRay = ray;
    shadowRay.TMax = lightTraceResult.m_hitT;
    TraceRay(rtScene, RAY_FLAG_FORCE_OPAQUE, RAY_TRACING_MASK_OPAQUE, RT_SHADOW_SHADER_INDEX, 1, 0, shadowRay, shadowRayPaylod);
    if(shadowRayPaylod.m_vHiTt > 0)
    {
        lightContribution = 0.0;
    }

    // ......

    radiance += lightContribution;
}
```

3.Combine the result with a multi-importance sampling weight.
```cpp
float previousCdfValue = index > 0 ? aLightPickingCdf[index - 1] : 0.0;
float lightPickPdf = (aLightPickingCdf[index] - previousCdfValue) / vLightPickingCdfPreSum;

lightContribution *= MISWeightRobust(materialSample.m_pdf,lightPickPdf * lightTraceResult.m_pdf);
```
## Store The Trace Result

Store the direct lighting result (bounce == 0) and the indirect lighting result together and accumulate the valid sample count. What's more, project the lighting luminance into the directionality SH. We will detail this store format in the latter "LightMap Encoding" part.

```cpp
{
    float TangentZ = saturate(dot(directLightRadianceDirection, worldFaceNormal));
    if(TangentZ > 0.0)
    {
        shDirectionality[rayIndex].rgba += Luminance(directLightRadianceValue) * SHBasisFunction(directLightRadianceDirection);
    }
    irradianceAndValidSampleCount[rayIndex].rgb += directLightRadianceValue;
}

{
    float TangentZ = saturate(dot(radianceDirection, worldFaceNormal));
    if(TangentZ > 0.0)
    {
        shDirectionality[rayIndex].rgba += Luminance(radianceValue) * SHBasisFunction(radianceDirection);
    }
    irradianceAndValidSampleCount[rayIndex].rgb += radianceValue;            
}

irradianceAndValidSampleCount[rayIndex].w += 1.0;
```
irradiance and count:
<p align="center">
    <img src="/resource/gi_rayguiding/image/irradiance_and_count.png" width="30%" height="30%">
</p>

directionality SH:
<p align="center">
    <img src="/resource/gi_rayguiding/image/directionality.png" width="30%" height="30%">
</p>


# First Bounce Ray Guiding (WIP)
>Note that ray guiding is disabled by default in our sample scene, since it still has some bugs.

Unreal GPULM has implemented a new algorithm called "first bounce ray guiding", which is similar to [<u>**[Ray Guiding For COD]**</u>](https://arisilvennoinen.github.io/Publications/ray_guiding_For_production_lightmap_baking_author_version.pdf).

## Path Guiding
A brief introduction to ralated algorithm (Path Guiding) referenced from Ray Guiding For COD:
>To improve the convergence of MC estimators, there has been a recent surge of work related to path guiding. The key idea is to collect samples of light transport and learn a probability distribution for importance sampling. Unlike analytical importance sampling methods, the guiding process is usually >designed in a way that it can take into account the light modulated **visibility term**, which is often impractical to tackle analytically in a closed form. 

## Ray Guiding Introduction

As we mentioned above, the single BRDF importance sampling estimate will have a large value due to p(f) being small. So we add additional light importance sampling with MIS:

>The standard importance sampling estimate {% katex %}\frac{f(x)g(x)}{p_{f}(x)}{% endkatex %}will have a very large value due to p(f) being small, and we will have high variance.

However, estimate lighting distribution is difficult. It's hard to deal with the visibility term unless you trace a visibility ray and compute the intersection with the scene, which is expensive for GI Baker. Furthermore, it's difficult to deal with secondary light sources.

Ray guiding in COD and Unreal GPULM runs in fixed memory footprint, independent of scene complexity. In addition, they are able to combine information from all the learning samples, leading to a more efficient estimator with reduced variance.

## Implementation

We have implemented part of the ray guiding algorithm. It contains four parts: clustering texels, building the guiding distribution, filtering the learned distribution and adjusting the sample PDF and direction.

We accumulate radiance in all directions during the first 128 light map pass tracing. Then, filter and build the ray guiding distribution based on the cluster radiance distribution. After that, adjust the sampling PDF and direction in the latter light map path tracing.

<p align="center">
    <img src="/resource/gi_rayguiding/image/ray_guiding_flow.png" width="50%" height="50%">
</p>

### Clustering Texels 

#### Concentric Mapping

At first, we generate the sample direction based on concentric mapping. Since the concentric mapping is an equal-area mapping, the PDF p(x,y) is constant, which allows us to sample uniformly with respect to area. Furthermore, we can perform inverse concentric mapping to map the sample direction into the 2D position in which the radiance result is stored to be used in the construction of the ray guiding distribution.

```cpp
float2 samplePoint = randomSample.yx;
float4 sampledValue = CosineSampleHemisphereConcentric(samplePoint);
float3 outDirection = TangentToWorld(sampledValue.xyz, N_World);

SMaterialSample materialSample = (SMaterialSample)0;
materialSample.m_direction = outDirection;
materialSample.m_pdf = outPdf;
```

<p align="center">
    <img src="/resource/gi_rayguiding/image/concentric_map.png" width="70%" height="70%">
</p>

We only perform concentric mapping on the first bounce.

```cpp
SMaterialSample materialSample;
#if USE_FIRST_BOUNCE_RAY_GUIDING == 1
if(bounce == 0)
{
    materialSample = SampleMaterial_RayGuiding(rtRaylod,randomSample,dispatch_thread_idx);
}
else
#endif
{
    materialSample  = SampleMaterial(rtRaylod,randomSample);
}
```

#### Cluster Texels

>To reduce variance in the learned ray distribution, we amplify the number of learning rays used to construct the PDF by sharing the PDF between clusters of spatially and geometrically coherent texels.
> ------ ray guiding for cod

In our example, every 8*8 pixels in the lightmap gbuffer share one cluster. The cluster size is 16 * 16, which corresponds to 256 sampling directions. The stored position in the cluster tile is a random sampling position before concentric mapping.

```cpp
int minRenderPassIndex = 0;
int maxRenderPassIndex = RAY_GUDING_MAX_SAMPLES;

if(( rtRenderPassInfo.m_renderPassIndex >= minRenderPassIndex) && (rtRenderPassInfo.m_renderPassIndex < maxRenderPassIndex))
{
    float2 Point = primaryRandSample.yx;
    float2 jittered_bin = Point * DIRECTIONAL_BINS_ONE_DIM;
    int2 position_in_bin = clamp(jittered_bin, float2(0, 0), float2(DIRECTIONAL_BINS_ONE_DIM - 1, DIRECTIONAL_BINS_ONE_DIM - 1));
    int2 final_position = ((rayIndex.xy / TEXEL_CLUSTER_SIZE)* DIRECTIONAL_BINS_ONE_DIM) + position_in_bin;

    float illuminance = luminance_first_bounce_ray_guiding * saturate(dot(radianceDirection, worldFaceNormal));
    InterlockedMax(rayGuidingLuminance[final_position], asuint(max(illuminance, 0)));
}
```
We store the maximum luminance for each direction using InterlockedMax. For positive floats we can cast them to uint and use atomic max directly.

<p align="center">
    <img src="/resource/gi_rayguiding/image/cluster_texels.png" width="70%" height="70%">
</p>

radiance distribution:

<p align="center">
    <img src="/resource/gi_rayguiding/image/radiance_distribution.png" width="50%" height="50%">
</p>

### Building & Filtering the Guiding Distribution

#### Filtering

>Even though we share all the texel samples within each cluster to learn the PDF, there is a risk of overfitting to the samples that can lead to slower convergence if the learned PDF does not reflect the population distribution well enough. To mitigate this effect, we apply a relaxation step by running a hierarchical smoothing filter over the quadtree representation of the guide distribution
> ------ ray guiding for cod

Filter the radiance results around the current pixel:

```cpp
float value = 0;
int filterKernelSize = 1;
for(int dx = -filterKernelSize; dx <= filterKernelSize; dx++)
{
	for(int dy = -filterKernelSize; dy <= filterKernelSize; dy++)
	{
        int2 final_pos = clamp(grp_idx.xy *  DIRECTIONAL_BINS_ONE_DIM + thread_idx.xy + int2(dx, dy), pixel_min_pos, pixel_max_pos);
        value += asfloat(rayGuidingLuminanceBuildSrc[final_pos]);
    }
}
```
#### Building Ray Guiding CDF

Build the CDF for each cluster. Compute the prefix sum per row using the WavePrefixSum operation. Then, obtain the sum of the row by using WaveReadLaneAt. It should be noticed that the compute group row size **MUST** be aligned to the warp size. Finally, calculate the normalized result based on the row sum.

<p align="center">
    <img src="/resource/gi_rayguiding/image/ray_guiding_cdfx.png" width="60%" height="60%">
</p>

The result computed above is ray guiding CDF in x dimension. We will compute the ray guiding CDF in y dimension next. Store the sum value for each row in the shared group during CDF X computation. Then, execute WaveReadLaneAt and WavePrefixSum to compute the CDF in the Y dimension.

```cpp
if (thread_idx.y == 0)
{
    if (thread_idx.x < DIRECTIONAL_BINS_ONE_DIM)
    {
	    float value = RowSum[thread_idx.x];
	    float prefixSum = WavePrefixSum(value) + value;
        float sum = WaveReadLaneAt(prefixSum, DIRECTIONAL_BINS_ONE_DIM - 1);

	    int2 writePos = int2(thread_idx.x % (DIRECTIONAL_BINS_ONE_DIM / 4), thread_idx.x / (DIRECTIONAL_BINS_ONE_DIM/ 4 ));
        int2 final_pos = grp_idx.xy * uint2(DIRECTIONAL_BINS_ONE_DIM / 4,DIRECTIONAL_BINS_ONE_DIM / 4) + writePos;

        float result = (sum == 0.0f) ? 0.0 : (prefixSum / sum);
	    rayGuidingCDFYBuildDest[final_pos] = result;
    }
}
```
Ray Guiding CDF X:
<p align="center">
    <img src="/resource/gi_rayguiding/image/CDFX.png" width="40%" height="40%">
</p>

Ray Guiding CDF Y:
<p align="center">
    <img src="/resource/gi_rayguiding/image/CDFY.png" width="40%" height="40%">
</p>

### Adjust the PDF

Adjust the PDF based on the ray guiding CDF x and y. Amplify the PDF if the luminance in the current direction contribute greatly to the final result.

```cpp
if(rtRenderPassInfo.m_renderPassIndex >= RAY_GUDING_MAX_SAMPLES)
{
    float lastRowPrefixSum = 0;
    float rowPrefixSum;
    int y = floor(primarySample.y * DIRECTIONAL_BINS_ONE_DIM);

    // ......
    // compute lastRowPrefixSum and rowPrefixSum
    pdfAdjust *= ((rowPrefixSum - lastRowPrefixSum) * DIRECTIONAL_BINS_ONE_DIM);

    float lastPrefixSum = 0;
	float prefixSum;
    int x = floor(primarySample.x * DIRECTIONAL_BINS_ONE_DIM);

    // ......
    // compute lastPrefixSum and prefixSum

    pdfAdjust *= ((prefixSum - lastPrefixSum) * DIRECTIONAL_BINS_ONE_DIM);
}

float out_pdf = ((NoL / PI) * pdfAdjust);
```
Adjust the sampling direction based on the radiance CDF during material importance sampling.

```cpp
if(rtRenderPassInfo.m_renderPassIndex >= RAY_GUDING_MAX_SAMPLES)
{
    float lastRowPrefixSum = 0;
	float rowPrefixSum = 0;

    // ......
    // compute the rowPrefixSum and lastRowPrefixSum
    samplePoint.y = float(y + (samplePoint.y - lastRowPrefixSum) / (rowPrefixSum - lastRowPrefixSum)) / DIRECTIONAL_BINS_ONE_DIM;

    float prefixSum = 0;
    float lastPrefixSum = 0;

    // ......
    // compute the prefixSum and lastPrefixSum
    samplePoint.x = float(x + (samplePoint.x - lastPrefixSum) / (prefixSum - lastPrefixSum)) / DIRECTIONAL_BINS_ONE_DIM;		
}
```
# Dilate

Because of bilinear interpolation and typical non-conservative rasterization, the sampling texels may blend with the background lightmap color. Furthermore, we may sample the texels that don't belong to the current mesh's light map since multiple meshes may share one lightmap atlas.

In our example, we solve this problem by expanding the lightmap with some padding board texels. Then, dilate the lightmap path tracing result. There is still a lot of work to be done on this simple implementation.

<p align="center">
    <img src="/resource/gi_rayguiding/image/dilate.png" width="100%" height="100%">
</p>


# Denoise

Unreal GPULM employs Odin(Open Image Denoise) to denoise the traced lightmap. We have implemented a custom denoising algorithm to replace the Odin, since HWRTL design goal is a single file hardware raytracing library and we don’t want to involve the thirdparty.

Our algorithm is based on [<u>**Non-Local Means filtering**</u>](https://www.cs.umd.edu/~zwicker/publications/AdaptiveRenderingNLM-SIGA12.pdf) combined with Joint Filtering.

## Bilateral Filtering

>A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels. This weight can be based on a Gaussian distribution. Crucially, the weights depend not only on Euclidean distance of pixels, but also on the radiometric differences (e.g., range differences, such as color intensity, depth distance, etc.). This preserves sharp edges
>------From Wikipedia

The bilateral filter is defined as:
{% katex %} w_{bf}(p,q)=exp(\frac{-||p-q||^{2}}{2\sigma^{2}_{s}})exp(\frac{-||c_{p}-c_{q}||^{2}}{2\sigma^{2}_{r}}){% endkatex %}<br><br>

A key issue with the bilateral filter, however, is that the range-distance estimation can be sensitive to noise.
## Non-Local Means

The non-local means (NL-Means) filter is a generalization of the bilateral filter where the range distance is computed using small patches {% katex %}P(x){% endkatex %}around the pixels [<u>**(NFOR)**</u>](https://benedikt-bitterli.me/nfor/nfor.pdf):
{% katex %} w_{nlm}(p,q)=exp(\frac{-||p-q||^{2}}{2\sigma^{2}_{s}})exp(\frac{-||P_{p}-P_{q}||^{2}}{k^{2}2\sigma^{2}_{r}}){% endkatex %}<br><br>
where k is a user parameter controlling the strength of the filter. The squared patch-based range distance is defined as:
{% katex %} ||P_{p}-P_{q}||^{2}=max(0,\frac{1}{|P|}\sum_{n \in P_{0}}d(p+n,q+n)){% endkatex %}<br><br>
where {% katex %}P_{p}{% endkatex %} is a square patch of size |P| (usually 7 x 7 pixels) centered on p, and P0 enumerates the offsets to the pixels within a patch. 
## Joint Filtering
If additional per-pixel information is available, the quality of bilateral or NL-Means filtering can be further improved by factoring the information into the weight-the so-called joint filtering[<u>**(NFOR)**</u>](https://benedikt-bitterli.me/nfor/nfor.pdf):
{% katex %}w_{jnlm}(p,q)=w_{nlm}(p,q)*w_{aux}(p,q){% endkatex %}
where {% katex %}w_{aux}(p,q){% endkatex %} is a weight derived from the additional information. Given a vector of k auxiliary feature buffers, {% katex %}f = (f1,..., fk){% endkatex %}, we can compute the weight as:
{% katex %}w_{aux}(p,q)=\Pi_{i=1}^{k}exp(-d_{f,i}(p,q)){% endkatex %}<br>
where:
{% katex %}d_{f,i}(p,q) = \frac{||f_{i,p}-f_{i,q}||^{2}}{2\sigma^{2}_{i,p}}{% endkatex %}<br>
and {% katex %}\sigma^{2}_{i}{% endkatex %} is the bandwidth of feature i. Joint filtering is typical for denoising MC renderings where feature buffers (e.g. normal, albedo, depth) can be obtained inexpensively as a byproduct of rendering the image

## Implementation

For the square patch Pp centered on p with the size of 20x20(HALF_SEARCH_WINDOW * 2) in our case, we compute the squared patch-based range distance to another patch Pq.

Pp search range:
```cpp
for (int searchY = -HALF_SEARCH_WINDOW; searchY <= HALF_SEARCH_WINDOW; searchY++) 
{
	for (int searchX = -HALF_SEARCH_WINDOW; searchX <= HALF_SEARCH_WINDOW; searchX++) 
    {
    }
}
```

Compute the color squared distance between these patches:
```cpp
for (int searchY = -HALF_SEARCH_WINDOW; searchY <= HALF_SEARCH_WINDOW; searchY++) 
{
	for (int searchX = -HALF_SEARCH_WINDOW; searchX <= HALF_SEARCH_WINDOW; searchX++) 
    {
        float2 searchUV = texUV + float2(searchX,searchY) * denoiseParamsBuffer.inputTexSizeAndInvSize.zw;
                
        float3 searchRGB = inputTexture.SampleLevel(gSamPointWarp, searchUV, 0.0).xyz;
        float3 searchNormal = denoiseInputNormalTexture.SampleLevel(gSamPointWarp, searchUV, 0.0).xyz;

        float patchSquareDist = 0.0f;
		for (int offsetY = -HALF_PATCH_WINDOW; offsetY <= HALF_PATCH_WINDOW; offsetY++) 
        {
			for (int offsetX = -HALF_PATCH_WINDOW; offsetX <= HALF_PATCH_WINDOW; offsetX++) 
            {
                float2 offsetInputUV = texUV + float2(offsetX,offsetY) * denoiseParamsBuffer.inputTexSizeAndInvSize.zw;
                float2 offsetSearchUV = searchUV + float2(offsetX,offsetY) * denoiseParamsBuffer.inputTexSizeAndInvSize.zw;

                float3 offsetInputRGB = inputTexture.SampleLevel(gSamPointWarp, offsetInputUV, 0.0).xyz;
                float3 offsetSearchRGB = inputTexture.SampleLevel(gSamPointWarp, offsetSearchUV, 0.0).xyz;
                float3 offsetDeltaRGB = offsetInputRGB - offsetSearchRGB;
                patchSquareDist += dot(offsetDeltaRGB, offsetDeltaRGB) - TWO_SIGMA_LIGHT_SQUARE;
			}
		}
    }
}
```

```cpp
float2 pixelDelta = float2(searchX, searchY);
float pixelSquareDist = dot(pixelDelta, pixelDelta);
weight *= exp(-pixelSquareDist / TWO_SIGMA_SPATIAL_SQUARE);
weight *= exp(-pixelSquareDist / FILTER_SQUARE_TWO_SIGMA_LIGHT_SQUARE);
```

Combine the result with the joint filter weight derived from the additional normal information:

```cpp
float3 normalDelta = inputNormal - searchNormal;
float normalSquareDist = dot(normalDelta, normalDelta);
weight *= exp(-normalSquareDist / TWO_SIGMA_NORMAL_SQUARE);
```

# LightMap Encoding

1.Store the sqrt irradiance result to give more precision in the draw region.

2.Separately store the integer and decimal parts of the log radiance.

3.Store the SH Directionality in a separate lightmap. The reason we store directionality separately instead of multiplying it with irradiance is that many pixels on the screen may share one texel and we can adjust the lighting result based on the world normal, which improves the visual quality.

```cpp
SEncodeOutputs EncodeLightMapPS(SEncodeGeometryVS2PS IN )
{
    SEncodeOutputs output;

    float4 lightMap0 = encodeInputIrradianceTexture.SampleLevel(gSamPointWarp, IN.textureCoord, 0.0).xyzw;
    float4 lightMap1 = encodeInputSHDirectionalityTexture.SampleLevel(gSamPointWarp, IN.textureCoord, 0.0).xyzw;

    float sampleCount = lightMap0.w;
    if(sampleCount > 0)
    {
        float4 encodedSH = lightMap1.yzwx;
        float3 irradiance = lightMap0.xyz / sampleCount;

        //todo: fixme
        const half logBlackPoint = 0.01858136;
        output.irradianceAndLuma = float4(sqrt(max(irradiance, float3(0.00001, 0.00001, 0.00001))), log2( 1 + logBlackPoint ) - (encodedSH.w / 255 - 0.5 / 255)); 
        output.shDirectionalityAndLuma = encodedSH;
    }
    else
    {
        output.irradianceAndLuma = float4(0,0,0,0);
        output.shDirectionalityAndLuma = float4(0,0,0,0);
    }

    return output;
}
```
LightMap Decoding:
```cpp
SVisualizeGIResult output;

float4 lightmap0 = visResultIrradianceAndLuma.SampleLevel(gSamPointWarp, IN.lightMapUV, 0.0);
float4 lightmap1 = visResultDirectionalityAndLuma.SampleLevel(gSamPointWarp, IN.lightMapUV, 0.0);

// irradiance
float3 irradiance = lightmap0.rgb * lightmap0.rgb;

// luma
float logL = lightmap0.w;
logL += lightmap1.w * (1.0 / 255) - (0.5 / 255);
const float logBlackPoint = 0.01858136;
float luma = exp2( logL ) - logBlackPoint;

// directionality
float3 wordlNormal = IN.normal;
float4 SH = lightmap1.xyzw;
float Directionality = dot(SH,float4(wordlNormal.yzx,1.0));

output.giResult = float4(irradiance * luma * Directionality, 1.0);
return output;
```

result:

<p align="center">
    <img src="/resource/gi_rayguiding/image/result.png" width="70%" height="70%">
</p>

[<u>**GI Baking Source Code**</u>](https://github.com/ShawnTSH1229/hwrtl)
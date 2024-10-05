---
title: >-
  [ReSTIR+SVGF-Sim]Spatiotemporal Reservoir Resampling With Spatiotemporal
  Filtering In MiniEngine
date: 2024-10-01 16:55:05
index_img: /resource/RestirSVGF/image/denoise_spec.png
tags:
---

# Introduction
We implement a GI algorithm combining ReSTIR and spatiotemporal filtering based on Unreal Engine and ReSTIR GI(SiGG 20). At first, we initial the reservoir samples using hardware raytracing at half resolution. Then we perform resampled importance sampling (RIS) to draw samples generated from sample importance resampling (SIR). To reduce time complexity O(M) and space complexity O(M), we use weighted reservoir sampling (WRS) to improve performance. After that, we reuse reservoir samples spatially and generate diffuse/specular lighting based on those samples. Finally, we perform temporal filter and spatial filter (joint bilateral filter) to denoise the indirect lighting results.
# [ReSTIR]Spatiotemporal Reservoir Resampling

<p align="center">
    <img src="/resource/RestirSVGF/image/ReSTIR_workflow.png" width="80%" height="80%">
</p>


## Theory

There are two basic problems in the Monte Carlo integration of rendering equation to reduce the errors and improve the sampling efficiency:

1.Given the limited sample number, find a distribution similar to the function f(x) in the integrand, making the samples more likely to be taken when the magnitude of the integrand is relatively large. In orther words, take the samples that are more importance.

2.How to draw random samples from the chosen probability distribution found in problem 1.

It's really hard to take a sample from the PDF that is proportional to {% katex %}L(w_i)f_r(w_i,w_o)cos(\theta_i){% endkatex %} for traditional sampling method (such as inverse sampling)，since it requires us to evaluate the integral at first, which becomes a chicken and egg problem. The sample importance resampling algorithm (SIR) described below solves this problem.

### Sample Importance Resampling (SIR)

#### SIR Workflow
Since drawing samples from the target PDF {% katex %}p_{target}(x){% endkatex %} proportional to {% katex %}L(w_i)f_r(w_i,w_o)cos(\theta_i){% endkatex %} is hard, SIR takes samples from a more simple PDF called proposal PDF {% katex %}p_{proposal}(x){% endkatex %}. Here is the basic workflow of the SIR:

1.Take M samples from proposal PDF {% katex %}p_{proposal}(x){% endkatex %}, named as {% katex %}X = x_1, x_2, ..., x_m{% endkatex %}. 
In our implementation, we use uniform sample hemisphere sampling to generate the samples, which means:
{% katex %}p_{proposal}(x) = \frac{1}{2\pi}{% endkatex %}<br><br>
2.For each sample taken from {% katex %}X{% endkatex %}, assign a weight for the sample, which is: 
{% katex %}w(x) = \frac{p_{target}(x)}{p_{proposal}(x)}{% endkatex %}. <br><br>
And 
{% katex %}p_{target}(x) = luminance(L(w_i)) {%endkatex %}<br><br>
3.Next, draw a sample from {% katex %}X{% endkatex %}, also named as target sample in this post, from this proposal sample set of M. The probability of each sample being picked is proportional to its weight, which means:
{% katex %} p(x_z|X) = \frac{w(x_z)}{\Sigma_{i=0}^{M}w(x_i)} {% endkatex %}<br><br>
The PDF of drawing a sample from the whole sample space is called SIR PDF. 
{% katex %}p_{sir}(x) = \frac{p_{target}(x)}{\frac{1}{M}\Sigma_{i=1}^{M}w(x_{i})}{% endkatex %}
#### [SIR PDF Derivation](https://www.zhihu.com/question/572528081/answer/3540624011)
Suppose the event {% katex %}Ez{% endkatex %} is sample {% katex %}z{% endkatex %} drawn from the sample set {% katex %}X{% endkatex %} and the event {% katex %}EX{% endkatex %} is sample {% katex %}z{% endkatex %} existing in the sample set {% katex %}X{% endkatex %}.<br><br>
{% katex %}p(Ez | EX) = \frac{p(Ez\cap EX)}{p(EX)}{% endkatex %}<br><br>
Where 
{% katex %}p(Ez | EX) = p(x_z|X) = \frac{w(x_z)}{\Sigma_{i=0}^{M}w(x_i)}{% endkatex %}<br><br>
And the SIR PDF is:
{% katex %} p_{sir}(x) = p(Ez\cap EX) {% endkatex %}<br><br>
{% katex %}p(Ez\cap EX) = p(Ez | EX) p(EX){% endkatex %}<br><br>
Draw a sample from the whole sample space, the probability of this sample being equal to {% katex %}z{% endkatex %} is {% katex %}p_{proposal}(z){% endkatex %}.  The probability that sample {% katex %}z{% endkatex %} exists in sample set {% katex %}X{% endkatex %} is:
{% katex %}p(EX) = p_{proposal}(z) + ...... + p_{proposal}(z) = p_{proposal}(z)*M{% endkatex %}<br><br>
Finally
{% katex %} p_{sir}(z) = p(Ez | EX) p(EX) {% endkatex %}<br><br>
{% katex %} = \frac{w(x_z)}{\Sigma_{i=0}^{M}w(x_i)} p(EX) {% endkatex %}<br><br>
{% katex %} = \frac{w(x_z)}{\Sigma_{i=0}^{M}w(x_i)} p_{proposal}(z)*M {% endkatex %}<br><br>
{% katex %} = \frac{p_{target}(z)}{\Sigma_{i=0}^{M}w(x_i)} *M {% endkatex %}<br><br>
{% katex %} = \frac{p_{target}(z)}{\frac{1}{M}\Sigma_{i=0}^{M}w(x_i)} {% endkatex %}<br><br>
When {% katex %}M = 1{% endkatex %},
{% katex %} p_{sir}(x) = \frac{p_{target}(x)}{w(x)} = p_{proposal}(z){% endkatex %}<br><br>
### Resampled Importance Sampling (RIS)

RIS uses importance sampling to draw samples from SIR. It should be noted that the SIR, similar to the inverse mapping, is a sampling distribution mapping instead of importance sampling !!! The RIS workflow is the same as Importance sampling:

1.Generate M samples by SIR as described in the SIR part.
2.Draw N samples from M samples generated by SIR using importance sampling from the SIR distribution.

The integration formula is shown below, which is the same as **importance sampling**:
{% katex %}I_{RIS}^{N,M}=\frac{1}{N}\Sigma_{i=1}^{N}\frac{f(x_{i})}{p_{sir}(x_{i})}{% endkatex %}<br><br>
Replace the SIR PDF with the formula derived before:
{% katex %}I_{RIS}^{N,M}=\frac{1}{N}\Sigma_{i=1}^{N}\frac{f(x_{i})}{p_{sir}(x_{i})}{% endkatex %}<br><br>
{% katex %}=\frac{1}{N}\Sigma_{i=1}^{N}(\frac{f(x_{i})}{p_{target}(x)}(\frac{1}{M}\Sigma_{p=0}^{M}w(x_{ip}))){% endkatex %}<br><br>
When N = 1, it becomes:
{% katex %}I_{RIS}^{1,M}=\frac{f(x_{i})}{p_{target}(x)}(\frac{1}{M}\Sigma_{p=0}^{M}w(x_{ip})){% endkatex %}<br><br>
,**Which is used in ReSTIR**.

###  Weighted Reservoir Sampling(WRS)

RIS requires us to pick a sample (N = 1) from M proposal samples generated by SIR, which causes two issues:

1.Requires O(M) space for storing proposal samples.
2.The time complexity of picking up a sample is O(M).

>Weighted reservoir sampling is an elegant algorithm that draws a sample from a stream of samples without pre-existed knowledge about how many samples are coming. Each sample in the stream would stand a chance that is proportional to its weight to be picked as the winner.

Assume we have n samples {% katex %}x_i\in[1,n]{% endkatex %}. Each sample weighs {% katex %}w(x_i){% endkatex %} and its probability is {% katex %}p(x_i){% endkatex %}. In ReSTIR, WRS **temporally** iterates the samples in the sample stream, picking and storing a sample randomly with the probability {% katex %}p(x_i){% endkatex %}.

{% katex %}p(x_i)=\frac{w(x_i)}{\Sigma_{j=1}^{n}w(x_j)}{% endkatex %}<br><br>

According to the above formula, WRS stores two variables:
Current picked sample({% katex %}w(x_i){% endkatex %})
Total processed weight({% katex %}\Sigma_{j=1}^{n}w(x_j){% endkatex %})

#### The Workflow of WRS

Assume we have iterated {% katex %}k-1(k\in[1,n]){% endkatex %}, the iterating result is stored in a structure named reservoir containing the following members:
1.The sample that is picked after (k-1) times of iteration.  {% katex %}y=x_t,t\in[1,k-1]{% endkatex %}
2.The sum of the weight.{% katex %}w_{sum} = \Sigma_{i=1}^{k-1}w(x_i){% endkatex %}

For an incoming sample in iteration k, we perform the following steps:
1.Generate a random number(r) between 0 and 1.
2.If {% katex %}r > \frac{w(x_k)}{w_{sum}+w(x_k)}{% endkatex %}, the incoming sample will be ignored.
3.If {% katex %}r <= \frac{w(x_k)}{w_{sum}+w(x_k)}{% endkatex %}, the new sample will be picked as the winner replacing the previously picked sample, if existed
4.Regardless, the total weight will be updated.{% katex %}w_{sum}=w_{sum}+w(x_k){% endkatex %}

#### Prof of Picking Probability

Assuming K samples have been processed, we want to prove that the probability of picking sample k-1 is euqal to the probability formula.

If sample K is ignored, the probability of sample K-1 being picked is {% katex %}p_1=\frac{w(x_{k-1})}{\Sigma_{i=1}^{k-1}w(x_i)}{% endkatex %}.
If sample K is picked, the probability of sample K being picked is p1, which means the probability of K-1 being ignored is {% katex %}p_2=\frac{w(x_{k})}{\Sigma_{i=1}^{k}w(x_i)}{% endkatex %}.

The probability of sample K-1 being picked when we process sample K is:
{% katex %}p(x_{k-1}) = p_1 * (1-p_2)= \frac{w(x_{k-1})}{\Sigma_{i=1}^{k-1}w(x_i)} * (1-\frac{w(x_{k})}{\Sigma_{i=1}^{k}w(x_i)}) {% endkatex %}<br><br>
{% katex %}= \frac{w(x_{k-1})}{\Sigma_{i=1}^{k-1}w(x_i)} * \frac{\Sigma_{i=1}^{k-1}w(x_i)}{\Sigma_{i=1}^{k}w(x_i)} {% endkatex %}<br><br>
{% katex %}= \frac{w(x_{k-1})}{\Sigma_{i=1}^{k}w(x_i)} {% endkatex %}<br><br>

## Implementation Detail
### Initial Sample

Our implementation is based on ReSTIR GI (SIGG 2022) and Unreal Engine, which provides a detailed description. 

<p align="center">
    <img src="/resource/RestirSVGF/image/algo_init.png" width="60%" height="60%">
</p>

>The first phase of our algorithm generates a new sample point for each visible point. Our implementation takes as input a G-buffer with the visible point’s position and surface normal in each pixel, though it could also easily be used with ray-traced primary visibiity

>For each pixel q with corresponding visible point xv, we sample a direction ωi using the source PDF pq(ωi) and trace a ray to obtain the sample point xs. The source PDF may be a uniform distribution, a cosine-weighted distribution, or a distribution based on the BSDF at the visible point.

Spatial-temporal blue noise (STBN) is useful for providing per-pixel random values to make noise patterns in renderings. It could generate a uniform random sequence in any direction, whether spatial or temporal. We use STBN to generate random rays due to its excellent properties.

```cpp
float2 E = GetBlueNoiseVector2(stbn_vec2_tex3d, reservoir_coord, g_current_frame_index);
float4 hemi_sphere_sample = UniformSampleHemisphere(E);

float3x3 tangent_basis = GetTangentBasis(world_normal);

float3 local_ray_direction = hemi_sphere_sample.xyz;
float3 world_ray_direction = normalize(mul(local_ray_direction, tangent_basis));
```

we use uniform sample hemisphere sampling to generate the samples, which means the proposal pdf is {% katex %}\frac{1}{2\pi}{% endkatex %}

To retrieve mesh data during ray tracing, we use bindless ray tracing to obtain hit point surface information. A mesh instance ID assigned during top-level acceleration structure construction.

```cpp
D3D12_RAYTRACING_INSTANCE_DESC& instanceDesc = instanceDescs[i];
......
instanceDesc.InstanceID = i;
......
```
Store the offsets of GPU resources, including index buffer offset, vertex attributes, material index and so on, for each mesh in a global byteaddress buffer.

```cpp
std::vector<RayTraceMeshInfo>   meshInfoData(model.m_Header.meshCount);
for (UINT i = 0; i < model.m_Header.meshCount; ++i)
{
    meshInfoData[i].m_indexOffsetBytes = model.m_pMesh[i].indexDataByteOffset;
    meshInfoData[i].m_uvAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[ModelH3D::attrib_texcoord0].offset;
    meshInfoData[i].m_normalAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[ModelH3D::attrib_normal].offset;
    meshInfoData[i].m_positionAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[ModelH3D::attrib_position].offset;
    meshInfoData[i].m_tangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[ModelH3D::attrib_tangent].offset;
    meshInfoData[i].m_bitangentAttributeOffsetBytes = model.m_pMesh[i].vertexDataByteOffset + model.m_pMesh[i].attrib[ModelH3D::attrib_bitangent].offset;
    meshInfoData[i].m_attributeStrideBytes = model.m_pMesh[i].vertexStride;
    meshInfoData[i].m_materialInstanceId = model.m_pMesh[i].materialIndex;
    ASSERT(meshInfoData[i].m_materialInstanceId < 27);
}
```
Retieval the surface attributes, including texture UV, vertex normal and so on, from the GPU resource based on the offset information stored on a prepared GPU buffer described above that is indexed by Instance ID.    

```cpp
        const float3 bary = float3(1.0 - attributes.x - attributes.y, attributes.x, attributes.y);

        RayTraceMeshInfo info = rt_scene_mesh_gpu_data[InstanceID()];
        const uint3 ii = Load3x16BitIndices(info.m_indexOffsetBytes + PrimitiveIndex() * 3 * 2);

        const float2 uv0 = asfloat(scene_vtx_buffer.Load2((info.m_uvAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes)));
        const float2 uv1 = asfloat(scene_vtx_buffer.Load2((info.m_uvAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes)));
        const float2 uv2 = asfloat(scene_vtx_buffer.Load2((info.m_uvAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes)));

        const float3 normal0 = asfloat(scene_vtx_buffer.Load3(info.m_normalAttributeOffsetBytes + ii.x * info.m_attributeStrideBytes));
        const float3 normal1 = asfloat(scene_vtx_buffer.Load3(info.m_normalAttributeOffsetBytes + ii.y * info.m_attributeStrideBytes));
        const float3 normal2 = asfloat(scene_vtx_buffer.Load3(info.m_normalAttributeOffsetBytes + ii.z * info.m_attributeStrideBytes));
```

Then sample the diffuse texture based on the retrieved UV calculated before and return those information to the RayGen shader.

```cpp
float3 vsNormal = normalize(normal0 * bary.x + normal1 * bary.y + normal2 * bary.z);

float2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
Texture2D<float4> diffTex = bindless_texs[info.m_materialInstanceId * 2 + 0];
const float3 diffuseColor = diffTex.SampleLevel(gSamLinearWarp, uv, 0).rgb;

float3 worldPosition = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

payload.world_normal = vsNormal;
payload.world_position = worldPosition;
payload.albedo = diffuseColor;
payload.hit_t = RayTCurrent();
```

Here is the structure of the sample in the ReSTIR GI paper.

<p align="center">
    <img src="/resource/RestirSVGF/image/sample_structure.png" width="75%" height="75%">
</p>

Our implementation is based on Unreal, which is different from the original paper:
1.There is no need to store the position and the normal of the visible point since they could be obtained from the GBuffer.
2.We store the ray direction and hit distance rather than hit position which could be calculated from the direction and hit distance.
3.We use STBN to store the random value, which means there is no need to store it.

```cpp
struct SSample
{
    float3 ray_direction;
    float3 radiance;
    float hit_distance;
    float3 hit_normal;
    float pdf;
};
```
Then, trace a shadow ray to determine whether the sample point is visible. If the sample point is visible, calculate the radiance.

```cpp
SInitialSamplingRayPayLoad shadow_payLoad = (SInitialSamplingRayPayLoad)0;
TraceRay(rtScene, RAY_FLAG_FORCE_OPAQUE, RAY_TRACING_MASK_OPAQUE, RT_SHADOW_SHADER_INDEX, 1,0, shadow_ray, shadow_payLoad);
if(shadow_payLoad.hit_t <= 0.0)
{
    float NoL = saturate(dot(light_direction.xyz, payLoad.world_normal));
    radiance = NoL * SUN_INTENSITY * payLoad.albedo * (1.0 / PI);
}
```

>After the fresh initial sample is taken, spatial and temporal resapling is applied. The target function {% katex %}p_{target}() = L_i(x_v,w_i) f(w_o,w_i)cos\theta = L_o(x_s,−w_i) f(wo,wi)cos\theta {% endkatex %} includes the effect of the BSDF and cosine factor at the visible point, though we have also found that the simple target function {% katex %}p_{target}() = L_i(x_v,w_i) {% endkatex %} **works well**. While it is a suboptimal target function for a single pixel, we have found that it is helpful for spatial resampling in that it preserves samples that may be effective at pixels other than the one that initially generated it.

The original paper ignore the BSDF and cos factor when calculate the tatget PDF. Since it "works well", we ignore those factors as well and use the luminance of the radiance as the target PDF.

```cpp
float target_pdf = Luminance(radiance);
float w = target_pdf / initial_sample.pdf;
```
After that, update the reservoir information. Since we only have one sample obtained from the ray tracing, meaning {% katex %} w_{new} / w_{sum} = 1{% endkatex %}, current sample will be stored in the reservoir absolutely. Below is the reservoir structure.

```cpp
struct SReservoir
{
    SSample m_sample;
    float weight_sum;
    float M; 
    float inverse_sir_pdf;
};
```
It contains 4 members:
1.m_sample: Reservoir sample picked up by WRS
2.weight_sum: the weight sum of the proposal samples being processed. (used for WRS)
3.M: the number of proposal samples.
4.inverse_sir_pdf: the inverse of the SIR PDF, which will be used for spatial sample reuse.

The SIR PDF derivation is in the SIR part above:
```cpp
reservoir.inverse_sir_pdf = reservoir.weight_sum / (max(reservoir.M * reservoir.m_sample.pdf, 0.00001f));
```
#### Result
Initial Sample Ray Direction:
<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_dir.png" width="75%" height="75%">
</p>

Initial Sample Ray Hit Distance:
<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_hit_dist.png" width="75%" height="75%">
</p>

Initial Sample Ray Hit Normal:
<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_normal.png" width="75%" height="75%">
</p>

Initial Sample Ray Hit Radiance:
<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_radiance.png" width="75%" height="75%">
</p>

Reservoir(Weight, M, Inverse SIR PDF):
<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_weights.png" width="75%" height="75%">
</p>



### Temporal Resampling
>After initial samples are generated, temporal resampling is applied. In this stage, for each pixel, we read the sample from the initial sample buffer, and use it to randomly update temporal reservoir,computing the RIS weight.

At first, reproject the sample into the previous frame.

```cpp
float4 pre_view_pos = mul(PreViewProjMatrix,float4(world_position, 1.0));
float2 pre_view_screen_pos = (float2(pre_view_pos.xy / pre_view_pos.w) * 0.5 + float2(0.5,0.5));
pre_view_screen_pos.y = (1.0 - pre_view_screen_pos.y);
uint2  pre_view_texture_pos = pre_view_screen_pos * g_restir_texturesize;
```
Then, check whether the sample is nearby the current sample based on world normal and world position.

```cpp
float noise = GetBlueNoiseScalar(stbn_scalar_tex, reservoir_coord, g_current_frame_index);
uint2 history_reservoir_sample = clamp(pre_view_texture_pos, uint2(0,0), int2(g_restir_texturesize) - int2(1,1));
float3 history_world_position = history_downsampled_world_pos[history_reservoir_sample].xyz;
float3 history_world_normal = history_downsampled_world_normal[history_reservoir_sample].xyz;
            
bool is_history_nearby = (distance(history_world_position, world_position) < 10.0f) && (abs(dot(history_world_normal,world_normal) > 0.25));
```
The original paper clearly describes what we should do in the next.

<p align="center">
    <img src="/resource/RestirSVGF/image/temp_reuse.png" width="75%" height="75%">
</p>

**4th line**: Assign a weight to the sample, as described in the SIR part above (step 2).
{% katex %}w(x) = \frac{p_{target}(x)}{p_{proposal}(x)}{% endkatex %}. <br><br>

The initial sample pass only generates one sample, which means {% katex %}w_{sum}=w_{new}{% endkatex %}. It is already stored in the InitialSampleBuffer (**2nd line**).

**5th line**: Update the reservoir, which pseudo code is shown below.
<p align="center">
    <img src="/resource/RestirSVGF/image/update_reservoir.png" width="70%" height="70%">
</p>

Corresponding code:
```cpp
bool UpdateReservoir(inout SReservoir reservoir, SSample new_sample, float weight_new, float noise)
{
    bool is_sample_changed = false;
    reservoir.weight_sum += weight_new;
    reservoir.M += 1.0f;

    if (noise < weight_new / reservoir.weight_sum)
	{
		reservoir.m_sample = new_sample;
		is_sample_changed = true;
	}
    return is_sample_changed;
}

bool MergeReservoirSample(inout SReservoir reservoir_sample, SReservoir new_reservoir_sample, float other_sample_pdf, float noise)
{
    float M0 = reservoir_sample.M;
    bool is_changed = UpdateReservoir(reservoir_sample, new_reservoir_sample.m_sample, other_sample_pdf * new_reservoir_sample.M * new_reservoir_sample.inverse_sir_pdf, noise);
    reservoir_sample.M = M0 + new_reservoir_sample.M;
    return is_changed;
}
```

The theory behind it is described in the WRS part. In fact, picking a sample from the proposal samples (the first part of the below figure) is the same as the WRS update method (the second part of the below figure) in **temporal**.

<p align="center">
    <img src="/resource/RestirSVGF/image/wrs_ris.png" width="65%" height="65%">
</p>

#### Result

Reservoir picked sample radiance(before temporal resuing VS after temporal reuse):

<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_radiance.png" width="75%" height="75%">
</p>

<p align="center">
    <img src="/resource/RestirSVGF/image/reservoir_ray_radiance_after_t_resuse.png" width="75%" height="75%">
</p>


### Spatial Resampling

Before talking about the real algorithm used in our implementation, we will describe the unbiased reservoir merge at first, which is not used due to performance costs.

<p align="center">
    <img src="/resource/RestirSVGF/image/unbiased_merge.png" width="65%" height="65%">
</p>

What we will be discussing is the fourth line. The weight used to combine two reservoirs is not the {% katex %}w_{sum}{% endkatex %} of other reservoirs, since the target PDF may be different from the current PDF.

For example, we use the radiance luminance as the target PDF. If the neighboring pixel's reservoir is not visible, its target PDF is 0. Assume the current pixel's reservoir is visible and merge the neighbor reservoir, It introduces bias.

As described in the above algorithm, for a neighbor reservoir prepared to merge, we use its proposal PDF but replace its target PDF with the current pixel's.

Assume we only consider the sample picked from the neighbor reservoir, which means the number of proposal sample in neighbor reservoir is 1. (M = 1).
In SIR part, We have proven that the SIR PDF is euqal to proposal PDF when M = 1.
{% katex %} p_{sir}(x) = \frac{p_{target}(x)}{w(x)} = p_{proposal}(x){% endkatex %}<br><br>

its weight equal to:

{% katex %} w(x_{neighbor}) = \frac{p_{target}(x_{neighbor})}{p_{proposal}(x_{neighbor})}{% endkatex %}<br><br>

Then, replace its target PDF with the current pixel’s.

{% katex %} w(x_{neighbor}) = \frac{p_{target}(x_{current})}{p_{proposal}(x_{neighbor})}{% endkatex %}<br><br>
{% katex %} = \frac{p_{target}(x_{current})}{p_{sir}(x_{neighbor})}{% endkatex %}<br><br>
{% katex %} = p_{target}(x_{current}) * inverse\_sir\_pdf{% endkatex %}<br><br>

However, there is still a difference in comparison with the original algorithm, which is M. In [this blog](https://agraphicsguynotes.com/posts/understanding_the_math_behind_restir_di/), the author explains it as a scaling factor. I don't really understand it. You can get more details on that blog if you are interested.  

<p align="center">
    <img src="/resource/RestirSVGF/image/biased_spatial.png" width="65%" height="65%">
</p>

Above is the real spatial algorithm we used in our implementation. The major difference is that the target PDF is not replaced with the current pixel's target PDF, which induces bias.

There are 3 methods to reduce the bias:
1.Geometric similarity test(6th line). if similarity is lower than the given threshold, skip it.
```cpp
bool is_neighbor_nearby = (distance(neighbor_world_position, world_position) < 10.0f) && (abs(dot(neighbor_world_normal,world_normal) > 0.6));
```
2.Sample the shadow map to calculate the visibility term, since the original paper points out that the bias is mainly in shadow areas due to visibility changes. We haven't implemented it.
<p align="center">
    <img src="/resource/RestirSVGF/image/visbility.png" width="85%" height="85%">
</p>
3.For spatial reservoirs, only operate on temporal reservoirs from nearby pixels and not spatial reservoirs, which is described in the original paper.

Another problems:

>With spatial reuse, it is necessary to account for differences in the source PDF between pixels that are due to the fact that our sampling scheme is based on the visible point’s position and surface normal.(Such a correction was not necessary in the original ReSTIR algorithm since it sampled lights directly without considering each pixel’s local geometry.) Therefore, when we reuse a sample from a pixel q at a pixel r, we must transform its solid angle PDF to current pixel’s solid angle space by dividing it by the Jacobian determinant of the corresponding transformation

```cpp
float CalculateJacobian(float3 world_position, float3 neighbor_world_position, SReservoir reservoir_sample)
{
    float3 neighbor_hit_position = neighbor_world_position + reservoir_sample.m_sample.hit_distance * reservoir_sample.m_sample.ray_direction;

    float3 neighbor_receiver_to_sample = neighbor_world_position - neighbor_hit_position;
    float original_distance = length(neighbor_receiver_to_sample);
    float original_cos_angle = saturate(dot(reservoir_sample.m_sample.hit_normal,neighbor_receiver_to_sample / original_distance));

    float3 new_receiver_to_sample = world_position - neighbor_hit_position;
    float new_distance = length(new_receiver_to_sample);
    float new_cos_angle = saturate(dot(reservoir_sample.m_sample.hit_normal,new_receiver_to_sample / new_distance));

    float jacobian = (new_cos_angle * original_distance * original_distance) / (original_cos_angle * new_distance * new_distance);

	if (isinf(jacobian) || isnan(jacobian)) { jacobian = 0; }
	if (abs(dot(reservoir_sample.m_sample.hit_normal, 1.0f)) < .01f) { jacobian = 1.0f; }
	if (jacobian > 10.0f || jacobian < 1 / 10.0f) { jacobian = 0; }
	return jacobian;
}
```
Reservoir picked sample radiance(before spatial resuing VS after spatial reuse):

<p align="center">
    <img src="/resource/RestirSVGF/image/before_sp_reused.png" width="65%" height="65%">
</p>
<p align="center">
    <img src="/resource/RestirSVGF/image/after_sp_reused.png" width="65%" height="65%">
</p>

### Upsample And Integrate

Below is the workflow of the ppsample and integrate pass:
1.Generate random upsalce samples using a spiral pattern.
```cpp
for(uint sample_index; sample_index < upscale_number_sample; sample_index++)
{
    const float angle = (sample_index + noise) * 2.3999632f;
    const float radius = pow(float(sample_index), 0.666f) * kernel_scale;

    const int2 reservoir_pixel_offset = int2(floor(float2(cos(angle), sin(angle)) * radius));
    uint2 reservior_sample_coord = clamp((current_pixel_pos / 2 + reservoir_pixel_offset) , uint2(0,0), int2(g_restir_texturesize) - int2(1,1));

    ......
}
```
2.Compute the diffuse light and specular light for current pixel.
```cpp
// diffuse lighting
float3 sample_light = reservoir_sample.m_sample.radiance * reservoir_sample.inverse_sir_pdf;
weighted_diffuse_lighting += sample_light * depth_weight * max(dot(world_normal, reservoir_sample.m_sample.ray_direction),0.0f);

// specular lighting
float3 H = normalize(view_direction + reservoir_sample.m_sample.ray_direction);
float NoH = saturate(dot(world_normal, H));
float D = D_GGX(0.8 * 0.8 ,NoH);
const float Vis_Implicit = 0.25;
weighted_specular_lighting += ToneMappingLighting(sample_light * D * Vis_Implicit) * depth_weight;

// weight
total_weight += depth_weight;
```

Diffuse Indirect Lighting(Scaled) and Specular Indirect Lighting(Scaled)

<p align="center">
    <img src="/resource/RestirSVGF/image/diff_indirect.png" width="65%" height="65%">
</p>
<p align="center">
    <img src="/resource/RestirSVGF/image/spec_indirect.png" width="65%" height="65%">
</p>


# Spatiotemporal Filtering
Following ReSTIR, we apply spatiotemporal filtering based on the UnrealEngine, which is similar to SVGF. If you want to dive deeper into SVGF, you can read my [Chinese blog](https://blog.csdn.net/u010669231/article/details/117885871) related to it.

## Temporal Filtering

Temporal filtering is simple. Reproject the sample into the previous frame and lerp between two frames. And we should perform a similarity test which is similar to TAA.

```cpp
float4 pre_view_pos = mul(PreViewProjMatrix,float4(world_position, 1.0));
float2 pre_view_screen_pos = (float2(pre_view_pos.xy / pre_view_pos.w) * 0.5 + float2(0.5,0.5));
pre_view_screen_pos.y = (1.0 - pre_view_screen_pos.y);

float2 histUV = pre_view_screen_pos;
float3 hist_diffuse = hist_diffuse_indirect.SampleLevel(defaultLinearSampler, histUV,0).xyz;
float3 hist_specular = hist_specular_indirect.SampleLevel(defaultLinearSampler, histUV,0).xyz;

float3 current_diffuse = output_diffuse_indirect[current_pixel_pos].xyz;
float3 current_specular = output_specular_indirect[current_pixel_pos].xyz;

//todo: temporal sample validation
output_diffuse_indirect[current_pixel_pos] = float4(lerp(hist_diffuse, current_diffuse, 0.05f),1.0);
output_specular_indirect[current_pixel_pos] = float4(lerp(hist_specular, current_specular, 0.05f),1.0);
```

## Spatial Filtering

In the spatial filter pass, we apply a joint bilateral filter pass to the result from the temporal filter pass. I would not detail it. If you are interested in it, my blog(GI Baking With Ray Guiding) decribes some denoise method including joint bilateral filter.  

```cpp
for(uint sample_index = 0; sample_index < biliteral_sample_num; sample_index++)
 {
     float2 offset = (Hammersley16(sample_index, biliteral_sample_num, random_seed) - 0.5f) * 2.0f * kernel_radius;
     int2 neighbor_position = current_pixel_pos + offset;
     if(all(neighbor_position < g_full_screen_texsize) && all(neighbor_position >= int2(0,0)))
     {
         float3 neighbor_world_position = gbuffer_world_pos[neighbor_position].xyz;
         if(any(neighbor_world_position != float3(0,0,0)))
         {
             float plane_distance = abs(dot(float4(neighbor_world_position, -1.0), current_sample_scene_plane));
             float relative_difference = plane_distance / sample_camera_distance;
             
             float depth_weight = exp2(-10000.0f * (relative_difference * relative_difference));
             
             float spatial_weight = exp2(-guassian_normalize* dot(offset, offset));

             float3 neighbor_normal = gbuffer_world_normal[neighbor_position].xyz;
             float angle_between_normal = acosFast(saturate(dot(world_normal, neighbor_normal)));
             float normal_weight = 1.0 - saturate(angle_between_normal * angle_between_normal);

             float sample_weight = spatial_weight * depth_weight * normal_weight;
             filtered_diffuse_indirect += ToneMappingLighting(input_diffuse_indirect[neighbor_position].xyz) * sample_weight;
             filtered_specular_indirect += ToneMappingLighting(input_specular_indirect[neighbor_position].xyz) * sample_weight;
             total_weight += sample_weight;
         }
     }
 }
```
Denoised Diffuse Indirect Lighting(Scaled) and Denoised Specular Indirect Lighting(Scaled)

<p align="center">
    <img src="/resource/RestirSVGF/image/denoise_diff.png" width="65%" height="65%">
</p>
<p align="center">
    <img src="/resource/RestirSVGF/image/denoise_spec.png" width="65%" height="65%">
</p>

# Result
Finally, combine the indirect lighting with the direct lighting.
```cpp
float3 simple_combine_indirect_light =  diffuse_indirect_color * diffuse * (1.0 / 3.1415926535) + sepcular_indirect_color * specular;
```
Enable ReSTIR VS Disable ReSTIR(Direct Lighting Only)

<p align="center">
    <img src="/resource/RestirSVGF/image/final_result.png" width="90%" height="90%">
</p>
<p align="center">
    <img src="/resource/RestirSVGF/image/direct_lighting_only.png" width="90%" height="90%">
</p>

[<u>**Project Source Code**</u>](https://github.com/ShawnTSH1229/restir)
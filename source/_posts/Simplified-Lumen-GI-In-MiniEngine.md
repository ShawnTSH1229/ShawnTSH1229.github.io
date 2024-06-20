---
title: Simplified Lumen GI In MiniEngine
date: 2024-05-18 13:47:44
index_img: /resource/simlumen/image/oct_result.png
categories:
- My Projects
---

# Introduction

This is a simplified Unreal Lumen GI implementation (**SimLumen**) based on **Unreal's Lumen GI**. We have implemented **most** of Unreal Lumen's features.

To perform fast ray tracing, SimLumen builds the **mesh SDFs** offline using the **embree** library. We also precompute a **global low resolution SDF** of the whole scene, which is used in surface cache ray tracing and screen probe voxel ray tracing.

SimLumen builds **mesh cards** offline in order to capture **material attributes** (normal, albedo) at run time. Mesh cards store the capture direction as well as the camera capture frustum. Since our meshes in the test example are simple and most are boxes, we generate only 6 cards for each mesh. Each direction corresponds a mesh card. At run time, SimLumen captures the mesh attributes, and copies them into a **global surface cache material attributes atlas**.

The surface cache describes the **lighting of the scene**. It contains 5 parts: **surface cache material attributes**, **surface cache direct lighting**, **surface cache indirect lighting**, **surface cache combined final lighting** and **voxelized scene lighting**.

With the global surface cache material attributes (normal, albedo and depth), SimLumen computes the direct lighting for each pixel in the surface cache atlas.

What's more, we have implemented **infinity** bounce lighting similar to Unreal Lumen. At first, we **voxelize** the scene. Each voxel has 6 directions. For each direction, we perform a mesh SDF trace and store the **hit mesh index and hit distance** in the **voxel visibility buffer**. Then, we **inject** the surface cache final lighting into the voxel if the voxel hit a mesh.

With the voxelized lighting, we compute the **surface cache indirect lighting** in the surface cache Atlas space. Firstly, SimLumen places probes every **4x4 pixels** in the Atlas space. In the next step, we trace the ray to the voxelized scene via global SDF and sample the radiance within the voxel. In order to denoise the trace result, SimLumen **filters** the radiance atlas and converts them into **spherical harmonics**. By integrating the probes around the pixel, we obtained surface cache indirect lighting.

The surface cache final lighting is computed by **combining surface direct and indirect lighting**.

As we have SDF to trace the scene quickly as well as surface cache that describes the scene lighting, we are able to proform the **screen space probe trace**.

SimLumen uses **importance sampling** to reduce the trace noisy. The PDF of the sampling function contains two parts: **BRDF PDF** and **lighting PDF**. The BRDF PDF is stored in **spherical harmonic** form, and we project the pixel BRDF PDF around the probe into spherical harmonics if the PDF is **not rejected** by the plane depth weight.  We use the **previous frame's** screen radiance result to estimate the lighting PDF by **reprojecting** the probe into the previous screen space, since we do not have information about the lighting source in the current frame. To improve performance, SimLumen employs **structured importance sampling** by reassigning the unimportant samples to those with a higher PDF.

Each probe traces 64 rays to the scene. SimLumen implements a **hybrid GI** similar to Unreal Lumen. The probes whose distance from the camera is less than 100 meters trace the scene by **mesh SDF** and sample the radiance from the **surface cache atlas**. Other probes use **Global SDF** to trace the scene and sample the radiance from the **voxel lighting**.

After that, we perform two additional passes to denoise the results. In the first pass, we **filter** the radiance with a uniform weight. Then, in the second pass, we convert the radiance into **spherical harmonics** and transform the SH into boardered **octchedron form**. This is usefull in hardware **bilinear sampling** in the following pass.

We finally obtained the final indirect lighting by integrating the probes around the current screen pixel and sampling the **octchedron form SH** by a linear sampler.


# Fast Ray Tracing

SimLumen uses signed distance fields to trace the ray with the mesh. SDF is a uniform mesh representation. What's more, it is differentiable, which means we can calculate the normal at the ray hit position. SimLumen precomputes the mesh SDF and the scene global SDF. GI nearby camera employs mesh SDF to accelerate ray-mesh intersection. Global SDF is used in distant GI and surface cache indirect lighting calculation.

## Build Mesh SDF Offline

The mesh SDF volume has different size based on the mesh bounding box. To pack these mesh SDF into a single 3D texture, we split the volume into bricks. Brick size is the same for mesh SDFs.

<p align="center">
    <img src="/resource/simlumen/image/build_mesh_sdf.drawio.png" width="60%" height="60%">
</p>

SimLumen use Embree to calculate the mesh SDF, which is the same as Unreal does. For each voxel, we traces 512 ray samples to intersect with the mesh and finds the closest hit position between these samples. A voxel is considered within the mesh if more than 25% of the 512 ray samples hit the triangle backface. We take the negative value of the closest hit distance.

```cpp
if (hit_num > 0 && hit_back_num > 0.25f * samples0.size())
{
	min_distance *= -1;
}
```

Then, scale and quantify the closest distance to uint8 format, and store the result into brick sdf data.

```cpp
float scaled_min_distance = min_distance / max_distance;// -1->1
float clamed_min_distance = Math::Clamp(scaled_min_distance * 0.5 + 0.5, 0, 1);//0 - 1
uint8_t normalized_min_distance = uint8_t(int32_t(clamed_min_distance * 255.0f + 0.5));

int brick_index = brick_index_z * volume_brick_num_y * volume_brick_num_x + brick_index_y * volume_brick_num_x + brick_index_x;
volumeData.distance_filed_volume[brick_index].m_brick_data[brick_vol_idx_x][brick_vol_idx_y][brick_vol_idx_z] = normalized_min_distance;
```

## Trace Mesh SDF

At runtime, we copy and upload each mesh's brick texture to the global mesh SDF brick texture and record the brick texture offset. We can find any SDF value at a given position in the mesh volume for each mesh by global brick texture and brick offset.

<p align="center">
    <img src="/resource/simlumen/image/build_scene_mesh_sdf.png" width="80%" height="80%">
</p>

The GI nearby camera uses mesh SDF to trace rays. In this case, we calculate the possible meshes that can be intersected in the ray tracing direction. Then, we perform SDF tracing for each mesh and find the closest hit position.

```cpp
[loop]
for(uint mesh_idx = 0; mesh_idx < scene_mesh_sdf_num; mesh_idx++)
{
    RayTraceSingleMeshSDF(world_position, ray_direction, 1000, mesh_idx, trace_result);
}
```

At first, SimNanite transform the ray from the world position into sdf volume position. If the ray intersects the volume bound box, we trace the ray from the intersection position. When the sphere trace step is over 64 or the closest distance in the sample position is closer than on voxel size, it means that we hit the mesh surface and should stop the sphere trace.

```cpp
float2 volume_space_intersection_times = LineBoxIntersect(volume_ray_start, volume_ray_end, volume_min_pos, volume_max_pos);

......
volume_space_intersection_times *= volume_max_trace_distance;

if((volume_space_intersection_times.x < volume_space_intersection_times.y) && (volume_space_intersection_times.x < trace_result.hit_distance))
{
    float sample_ray_t = volume_space_intersection_times.x;

    uint max_step = 64;
    bool bhit = false;
    uint step_idx = 0;

    [loop]
    for( ; step_idx < max_step; step_idx++)
    {
        float3 sample_volume_position = volume_ray_start + volume_ray_direction * sample_ray_t;
        float distance_filed = SampleDistanceFieldBrickTexture(sample_volume_position, mesh_sdf_info);
        float min_hit_distance = mesh_sdf_info.volume_brick_size * 0.125 * 1.0; // 1 voxel

        if(distance_filed < min_hit_distance)
        {
            bhit = true;
            sample_ray_t = clamp(sample_ray_t + distance_filed - min_hit_distance, volume_space_intersection_times.x, volume_space_intersection_times.y);
            break;
        }

        sample_ray_t += distance_filed;

        if(sample_ray_t > volume_space_intersection_times.y + min_hit_distance)
        {
            break;
        }
    }

    if(step_idx == max_step)
    {
        bhit = true;
    }

    if(bhit && sample_ray_t < trace_result.hit_distance)
    {
        trace_result.is_hit = true;   
        trace_result.hit_distance = sample_ray_t;
        trace_result.hit_mesh_index = object_index;
        trace_result.hit_mesh_sdf_card_index = mesh_sdf_info.mesh_card_start_index;
    }
}
```
## SDF Normal

We can calculate the normal at any sample position by calculating the closest distance gradient of the voxels around it.

<p align="center">
    <img src="/resource/simlumen/image/sdf_normal.png" width="70%" height="70%">
</p>

```cpp
float3 CalculateMeshSDFGradient(float3 sample_volume_position, SMeshSDFInfo mesh_sdf_info)
{
    float voxel_offset = mesh_sdf_info.volume_brick_size * 0.125;

    float R = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x + voxel_offset, sample_volume_position.y, sample_volume_position.z),mesh_sdf_info);
    float L = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x - voxel_offset, sample_volume_position.y, sample_volume_position.z),mesh_sdf_info);

    float F = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x, sample_volume_position.y + voxel_offset, sample_volume_position.z),mesh_sdf_info);
    float B = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x, sample_volume_position.y - voxel_offset, sample_volume_position.z),mesh_sdf_info);

    float U = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x, sample_volume_position.y, sample_volume_position.z + voxel_offset),mesh_sdf_info);
    float D = SampleDistanceFieldBrickTexture(float3(sample_volume_position.x, sample_volume_position.y, sample_volume_position.z - voxel_offset),mesh_sdf_info);

    float3 gradiance = float3(R - L, F - B, U - D);
	return gradiance;
}
```
Below is a SDF ray-tracing visualization.  In this example, the ray-tracing direction for each cube is determined by its offset from the center cube. RGB colors represent the hit position's normal. X is represented by red color, Y by green color, and Z by blue color.

<p align="center">
    <img src="/resource/simlumen/image/sdf_normal_visualize.png" width="80%" height="80%">
</p>


## Global SDF

The global signed distance field is a low resolution SDF of the whole scene. We precompute the global SDF offline. Global SDF differs from scene mesh SDF in that the scene mesh SDF is stored in brick textures with fixed z-dimensions, whereas global SDF is stored in a size-scalable volume texture based on the scene bounding box.

# Surface Cache

The mesh SDF trace allows us to determine the hit position of a ray, however it does not provide information regarding the material attributes (albedo, normal, etc.) at the hit position. Unreal Lumen uses mesh cards to capture these material attributs runtime. 

Mesh cars describe the material attributes capture infomation. It can be generated offline. However, material attributes capture must be performed at runtime, since occlusion between scene meshes can't be determined offline.

## SimLumen Card Generation

**In Unreal, a mesh may have many mesh cards depend on mesh complexity.** SimLumen simplifies mesh card generation: generates a fixed number mesh card (6 direction) based on the mesh bounding box.  

<p align="center">
    <img src="/resource/simlumen/image/mesh_card.png" width="40%" height="40%">
</p>

We calculate the capture position and direction from the bounding box directly. The mesh card depth is determined by software raytracing. For each texel in the mesh card, we trace a ray in the mesh card direction, calculate the ray-mesh intersection and find the furthest intersection distance.
```cpp
	float max_depth = 2.0;

	for (int x_idx = 0; x_idx < 128; x_idx++)
	{
		for (int y_idx = 0; y_idx < 128; y_idx++)
		{
			XMFLOAT3 trace_position = palne_start_trace_pos;
			AddFloatComponent(trace_position, dimension_x, (x_idx + 0.5) * x_stride);
			AddFloatComponent(trace_position, dimension_y, (y_idx + 0.5) * y_stride);

			RTCRayHit embree_ray;
			......
			rtcIntersect1(m_rt_scene, &embree_ray, &args);

			if ((embree_ray.ray.tfar != 1e30f) && embree_ray.hit.geomID != RTC_INVALID_GEOMETRY_ID && embree_ray.hit.primID != RTC_INVALID_GEOMETRY_ID)
			{
				Math::Vector3 hit_normal(embree_ray.hit.Ng_x, embree_ray.hit.Ng_y, embree_ray.hit.Ng_z);
				hit_normal = Math::Normalize(hit_normal);
				float dot_value = Math::Dot(trace_dir, hit_normal);

				if (dot_value < 0 && max_depth < embree_ray.ray.tfar)
				{
					max_depth = embree_ray.ray.tfar;
				}
			}
		}
	}

	XMFLOAT3 points[4];
	points[0] = palne_start_trace_pos;
	points[1] = plane_end_trace_pos;
	
	points[2] = palne_start_trace_pos;
	points[2].x += trace_dir.x * max_depth;
	points[2].y += trace_dir.y * max_depth;
	points[2].z += trace_dir.z * max_depth;

	points[3] = plane_end_trace_pos;
	points[3].x += trace_dir.x * max_depth;
	points[3].y += trace_dir.y * max_depth;
	points[3].z += trace_dir.z * max_depth;

	Math::BoundingBox RetBound;
	Math::BoundingBox::CreateFromPoints(RetBound, 4, points, sizeof(XMFLOAT3));
	return RetBound;
```

## Material Attributes Capture

SimLumen captures the mesh card at runtime. After the mesh card capture is completed, we copy these mesh card attribute textures into a global card atlas.

<p align="center">
    <img src="/resource/simlumen/image/mat_attributes_alebedo.png" width="35%" height="35%">
</p>
<p align="center">
    <img src="/resource/simlumen/image/mat_attributes_normal.png" width="35%" height="35%">
</p>

Below is a visualization of the mesh card's normal and albedo for a scene:

<p align="center">
    <img src="/resource/simlumen/image/scene_attributes_alebedo.png" width="50%" height="50%">
</p>
<p align="center">
    <img src="/resource/simlumen/image/scene_attributes_normal.png" width="50%" height="50%">
</p>

## Surface Cache Lighting

Unreal Lumen has implemented an **infinity bounce** lighting by the combination of the surface cache lighting and voxel lighting. Here is the surface cache flow graph:

Step 1: Calculate direct lighting with surface cache attributes (albedo / normal).

Step 2: Combine direct lighting and indirect lighting together. In the first frame, direct lighting results in black, since no light has been injected into the scene voxel.

Step 3: Inject the combined light into the scene voxel.

Step 4: Calculate indirect lighting used in the next frame.

<p align="center">
    <img src="/resource/simlumen/image/surface_cache_lighting_flow.png" width="75%" height="75%">
</p>

## Surface Cache Direct Lighting

The world position of a mesh card pixel is calculated by the card rotation matrix and card depth. 

```cpp
float3 local_position;
local_position.xy = (card_uv * (2.0f) - 1.0f) * card_info.rotated_extents.xy;
local_position.z = -(depth * 2.0 - 1.0f)  * card_info.rotated_extents.z;
float3 rotate_back_pos = mul((float3x3)card_info.rotate_back_matrix, local_position);
rotate_back_pos += card_info.bound_center;;
```

After that, SimLumen transforms the world position into shadow space in order to determine whether the pixel is inside or outside the shadow.
```cpp
float shadow = 0.0;
{
    float4 shadow_screen_pos = mul(ShadowViewProjMatrix, float4(card_data.world_position,1.0));
    float2 shadow_uv = shadow_screen_pos.xy;
    shadow_uv = shadow_uv * float2(0.5, -0.5) + float2(0.5, 0.5);
    float2 shadow_pixel_pos = shadow_uv.xy * 2048;

    float shadow_depth_value = shadow_depth_buffer.Load(int3(shadow_pixel_pos.xy,0)).x;;
    shadow = ((shadow_screen_pos.z + 0.0005) < shadow_depth_value ) ? 0.0 :1.0;
}
```
We calculate the direct lighting for each light source and accumulate them if the scene has many light sources.
```cpp
float3 directional_lighting = float3(0,0,0);
{
    float3 light_direction = SunDirection;
    float NoL = saturate(dot(light_direction, card_data.world_normal));
    directional_lighting = SunIntensity * NoL * card_data.albedo * shadow;
}

float3 point_lighting = float3(0,0,0);
{
    float3 point_light_direction = point_light_world_pos - card_data.world_position;
    float3 light_dist = length(point_light_direction);
    float attenuation = saturate((point_light_radius - light_dist) / point_light_radius);   
    float NoL = saturate(dot(normalize(point_light_direction), card_data.world_normal));
    point_lighting = NoL * card_data.albedo * attenuation * attenuation;
}
surface_cache_direct_lighting[int2(pixel_pos.xy)] = float4(point_lighting + directional_lighting, 1.0);
```

surface cache direct lighting visualization:
<p align="center">
    <img src="/resource/simlumen/image/scache_direct_lighting.png" width="64%" height="64%">
</p>

## Voxel Visibility Buffer

Voxel visibility buffer stores the hit mesh index and hit distance in x/y/z direction. As a persistent data, it is only updated when the meshes' positions change. SDF traces are performed for each voxel along the xyz direction for meshes that are possibly intersected with its center. If the ray from the voxel center along the x/y/z direction hit a mesh, we store the mesh index in the voxel. This will be used in the next light injection pass.

<p align="center">
    <img src="/resource/simlumen/image/voxel_vis_info.png" width="60%" height="60%">
</p>

## Lighting Injection

We can obtain the intersection mesh and intersection position directly from the voxel visibility buffer. After that, we transform the hit world position into mesh card space location and calculate the final light atlas UV. The final light is the combination of the direct light and indirect light. However, the indirect lighting remains black until the second frame as it is dependent on voxel illumination.

```cpp
    uint direction_idx = group_idx.y;
    SVoxelVisibilityInfo voxel_vis_info = scene_voxel_visibility_buffer[voxel_index_1d];
    int mesh_index = voxel_vis_info.voxel_vis_info[direction_idx].mesh_index;
    if(mesh_index != -1)
    {
        SMeshSDFInfo mesh_info = scene_sdf_infos[mesh_index];
        uint card_index = direction_idx; 
        uint global_card_index = mesh_info.mesh_card_start_index + card_index;

        SCardInfo card_info = scene_card_infos[global_card_index];

        float hit_distance = voxel_vis_info.voxel_vis_info[direction_idx].hit_distance;
        float3 light_direction = voxel_light_direction[direction_idx];
        float3 hit_world_pos = voxel_world_pos + light_direction * hit_distance;

        float2 uv = GetCardUVFromWorldPos(card_info, hit_world_pos);

        uint2 card_index_xy = uint2((global_card_index % card_num_xy), (global_card_index / card_num_xy));
        uint2 pixel_pos = card_index_xy * 128 + float2(uv * 128);
        pixel_pos.y = SURFACE_CACHE_TEX_SIZE - pixel_pos.y;
        float3 final_lighting = final_lighting_tex.Load(int3(pixel_pos,0)).xyz;

        scene_voxel_lighting[voxel_index_1d].final_lighting[direction_idx] = final_lighting;
    }
```

voxel lighting visualization:

<p align="center">
    <img src="/resource/simlumen/image/voxel_lighting_visualize.png" width="60%" height="60%">
</p>

## Surface Cache Indirect Lighting

The first step is to calculate the radiance of the current surface cache pixel using global SDF, filter, and store the radiance in a radiance atlas. We then convert radiance into SH, which allows us to perform probe interpolation to reduce lighting noise.
The irradiance in a unit area is the integral of the f(x) over the half sphere:
{% katex %}E(p) = \int_{\Omega}L(p,i)max(0,n\cdot i )\mathrm{d}i{% endkatex %}<br>
F(x) can be split into two parts: the lighting function and the diffuse transfer function:
{% katex %}E(p) = \int_{\Omega}L(i)\cdot H(i)\mathrm{d}i{% endkatex %}<br>
L(i) is reconstructed from the spherical harmonic by SH factors and biasis function:
{% katex %}L(i) \approx \Sigma l_{k}B_{k}(i) {% endkatex %}<br>
We project the radiance into the basis function to get the SH factors in Convert SH pass:
{% katex %}l_{k} = \int_{\Omega}L(i)B(i)\mathrm{d}i{% endkatex %}<br>
```cpp
irradiance_sh = AddSH(irradiance_sh, MulSH(SHBasisFunction(world_ray), trace_irradiance / pdf));
```

If we project both the illumination and transfer functions into SH coefficients then orthogonality guarantees that the integral of the function's products is the same as the dot product of their coefficients:
{% katex %}E(p) = \Sigma_{k=0}^{n^{2}} l_{k}h_{k}{% endkatex %}<br>
{% katex %}H(i) = max(0,n\cdot l) {% endkatex %}<br>
The irradiance of a pixel is calculated in the integrate pass:
```cpp
FTwoBandSHVector diffuse_transfer_sh = CalcDiffuseTransferSH(card_data.world_normal, 1.0f);
float3 texel_irradiance = max(float3(0.0f, 0.0f, 0.0f), DotSH(irradiance_sh, diffuse_transfer_sh));
```


<p align="center">
    <img src="/resource/simlumen/image/scache_indirect_lighting.png" width="40%" height="40%">
</p>

### Radiosity Trace

SimLumen split the atlas space surface cache into 8x8 tiles. Each tile place 2x2 probes. 

<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_trace_probe.png" width="30%" height="30%">
</p>

Each probe performs 16 ray tracings in the hemisphere direction.

<p align="center">
    <img src="/resource/simlumen/image/radiosity_probe_ray.png" width="30%" height="30%">
</p>

To accelerate the speed of convergence of the integrate, the probe center is jittered according to the tile index and the frame index.
```cpp
uint2 probe_jitter = GetProbeJitter(indirect_lighting_temporal_index);
```

For each direction, we trace a ray from the probe center and find the world space hit position by global SDF.

```cpp
float3 world_ray;
float pdf;
GetRadiosityRay(tile_idx, sub_tile_pos, card_data.world_normal, world_ray, pdf);

SGloablSDFHitResult hit_result = (SGloablSDFHitResult)0;
TraceGlobalSDF(card_data.world_position + card_data.world_normal * gloabl_sdf_voxel_size * 2.0, world_ray, hit_result);
```
Finally, fetch the voxel lighting at the hit position and accumulate the weighted lighting results in the x/y/z direction.

```cpp
            uint voxel_index_1d = GetVoxelIndexFromWorldPos(hit_world_position);

            SVoxelLighting voxel_lighting = scene_voxel_lighting[voxel_index_1d];
            float3 voxel_lighting_x = voxel_lighting.final_lighting[x_dir];
            float3 voxel_lighting_y = voxel_lighting.final_lighting[y_dir];
            float3 voxel_lighting_z = voxel_lighting.final_lighting[z_dir];

            float weight_x = saturate(dot(world_ray, voxel_light_direction[x_dir]));
            float weight_y = saturate(dot(world_ray, voxel_light_direction[y_dir]));
            float weight_z = saturate(dot(world_ray, voxel_light_direction[z_dir]));

            radiance += voxel_lighting_x * weight_x;
            radiance += voxel_lighting_y * weight_y;
            radiance += voxel_lighting_z * weight_z;

            radiance /= (weight_x + weight_y + weight_z);
```

radiance trace result:
<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_atlas.png" width="60%" height="60%">
</p>

radiance trace visualization:
<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_vis.png" width="60%" height="60%">
</p>

### Radiosity Filter
In this pass, SimLumen filters the radiance atlas to reduce the noise. We sample the radiance around the current texel and accumulate weighted samples. Radiance sample weights in Unreal Lumen are dependent upon a number of factors, including the texel's World space plane and the distance between the planes.

filtered radiance atlas:

<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_atlas_filtered.png" width="60%" height="60%">
</p>

### Convert To SH
Radiance atlas results are still noisy after filtering, since we only have 16 samples per probe. We solve this problem by converting the tile radiance into two bands SH, which allows us to interpolate the probes more easily. 
```cpp
for(uint trace_idx_x = 0; trace_idx_x < SURFACE_CACHE_PROBE_TEXELS_SIZE; trace_idx_x++)
{
    for(uint trace_idx_y = 0; trace_idx_y < SURFACE_CACHE_PROBE_TEXELS_SIZE; trace_idx_y++)
    {
        ......
        float3 trace_irradiance = trace_radiance_atlas.Load(int3(pixel_atlas_pos.xy, 0)).xyz;
        irradiance_sh = AddSH(irradiance_sh, MulSH(SHBasisFunction(world_ray), trace_irradiance / pdf));
		num_valid_sample += 1.0f;
    }
}

if (num_valid_sample > 0)
{
	irradiance_sh = MulSH(irradiance_sh, 1.0f / num_valid_sample);
}
```
### Radiosity Integrate

Finally, sample the probes around the current pixel and calculate the weights based on the atlas position. Then, accumulate the SH weights and weighted SH, calculate the basis function using the current pixel's world normal, and dot product the basis function with the SH result.  By dividing it by the total sum of SH weights, we get the final radiance value for the current pixel.

```cpp
        FTwoBandSHVectorRGB irradiance_sh = (FTwoBandSHVectorRGB)0;

        FTwoBandSHVectorRGB sub_irradiance_sh00 = GetRadiosityProbeSH(ProbeCoord00);
        FTwoBandSHVectorRGB sub_irradiance_sh01 = GetRadiosityProbeSH(ProbeCoord01);
        FTwoBandSHVectorRGB sub_irradiance_sh10 = GetRadiosityProbeSH(ProbeCoord10);
        FTwoBandSHVectorRGB sub_irradiance_sh11 = GetRadiosityProbeSH(ProbeCoord11);

        irradiance_sh = AddSH(irradiance_sh, MulSH(sub_irradiance_sh00, weights.x));
        irradiance_sh = AddSH(irradiance_sh, MulSH(sub_irradiance_sh01, weights.y));
        irradiance_sh = AddSH(irradiance_sh, MulSH(sub_irradiance_sh10, weights.z));
        irradiance_sh = AddSH(irradiance_sh, MulSH(sub_irradiance_sh11, weights.w));

        uint card_index_1d = card_idx_2d.y * SURFACE_CACHE_CARD_NUM_XY + card_idx_2d.x;
        SCardInfo card_info = scene_card_infos[card_index_1d];
        SCardData card_data = GetSurfaceCardData(card_info, float2(uint2(thread_index.xy % 128u)) / 128.0f, pixel_atlas_pos.xy);
        FTwoBandSHVector diffuse_transfer_sh = CalcDiffuseTransferSH(card_data.world_normal, 1.0f);
        
        float3 texel_irradiance = max(float3(0.0f, 0.0f, 0.0f), DotSH(irradiance_sh, diffuse_transfer_sh));
        irtexel_radiance = texel_irradiance / (weights.x + weights.y + weights.z + weights.w);
```

surface cache indirect lighting visualization:
<p align="center">
    <img src="/resource/simlumen/image/indirect_lighting_vis.png" width="60%" height="60%">
</p>

surface cache combined lighting visualization:
<p align="center">
    <img src="/resource/simlumen/image/combined_lighting_vis.png" width="60%" height="60%">
</p>

# Final Gather

We place the probe in screen space for each 8x8 pixels and use octchedron mapping to map the screen coordinates into spherical coordinates. Each probe trace 64 rays into the scene.

## Importance Sampling

It's too noisy if we employ uniform sampling rather than importance sampling. 
without importance sampling Vs with importance sampling:

<p align="center">
    <img src="/resource/simlumen/image/is_vs_no_is.png" width="60%" height="60%">
</p>

What we do in the importance sampling part is searching the rays that orientates to the lighting source and world normal. That is to say, we peroform importance sampling for BRDF(fs) term and input radiance(Li) term:
{% katex %}\frac{1}{N}\Sigma_{k=1}^{N}\frac{L_{i}(l)f_{s}(l->v)cos(\theta l)}{P_{k}}{% endkatex %}<br>

### BRDF PDF

In this step, SimLumen generate the three band sphere harmonic factors for the BRDF function. We sample the screen pixels around the screen probe and compute the influence weight on the probe. If the weight is over the threshold, convert the BRDF to SH and accumulate the SH. Then write the result to the BRDF SH buffer.

```cpp
            float3 pixel_world_position = GetWorldPosByDepth(thread_depth, piexl_tex_uv);
            float3 probe_world_position = GetWorldPosByDepth(probe_depth, ss_probe_atlas_pos / global_thread_size);

            float4 pixel_world_plane = float4(thread_world_normal, dot(thread_world_normal,pixel_world_position));
            float plane_distance = abs(dot(float4(probe_world_position, -1), pixel_world_plane));

            float probe_view_dist = length(probe_world_position - CameraPos);
            float relative_depth_diff = plane_distance / probe_view_dist;
            float depth_weight = exp2(-10000.0f * (relative_depth_diff * relative_depth_diff));
            if(depth_weight > 0.1f)
            {
                uint write_index;
                InterlockedAdd(group_num_sh, 1, write_index);

                FThreeBandSHVector brdf = CalcDiffuseTransferSH3(thread_world_normal, 1.0);
                WriteGroupSharedSH(brdf, write_index);
            }
```

The pixel normal may be located in a different plane from the probe. Therefore, we compute the plane weight for the given pixel and reject the pixel if the depth weight is over the threshold. Then, store the results of those valid pixels in a **group shared** array.

<p align="center">
    <img src="/resource/simlumen/image/brdf_depth_weight.png" width="60%" height="60%">
</p>

After that, perform a parallel reduction to accumulate these SH factors.

<p align="center">
    <img src="/resource/simlumen/image/acc_brdf_pdf.png" width="60%" height="60%">
</p>

Finally, the first nine threads store the 9 SH factors in output BRDF SH buffer.
```cpp
        if (thread_index < 9 && group_num_sh > 0)
        {
            uint write_index = (ss_probe_idx_xy.y * screen_probe_size_x + ss_probe_idx_xy.x) * 9 + thread_index;
            float normalize_weight = 1.0f / (float)(group_num_sh);
            brdf_pdf_sh[write_index] = pdf_sh[offset][thread_index] * normalize_weight;
        }
```

brdf pdf visualization:
<p align="center">
    <img src="/resource/simlumen/image/brdf_pdf_vis.png" width="60%" height="60%">
</p>

### Lighting PDF

The light source direction in the current frame is unknown. In order to search the light direction, we assume lighting changes slightly and reuse the previous frame's lighting result.

<p align="center">
    <img src="/resource/simlumen/image/ss_light_is.png" width="60%" height="60%">
</p>

Then, reproject the probe into the previous frame screen position and find the corresponding direction texel.

Calculate the lighting pdf based on its luminance.

```cpp
            const float2 global_thread_size = float2(is_pdf_thread_size_x,is_pdf_thread_size_y);
            float3 probe_world_position = gbuffer_c.Load(int3(ss_probe_atlas_pos.xy,0)).xyz;
            
            float4 pre_view_pos = mul(PreViewProjMatrix,float4(probe_world_position, 1.0));
            float2 pre_view_screen_pos = (float2(pre_view_pos.xy / pre_view_pos.w) * 0.5 + float2(0.5,0.5));
            pre_view_screen_pos.y = (1.0 - pre_view_screen_pos.y);
            pre_view_screen_pos = pre_view_screen_pos * global_thread_size;
            uint2 pre_probe_pos = uint2(pre_view_screen_pos) / uint2(PROBE_SIZE_2D,PROBE_SIZE_2D);
            uint2 pre_texel_pos = pre_probe_pos * uint2(PROBE_SIZE_2D,PROBE_SIZE_2D) + group_thread_idx.xy;

            lighting = sspace_composited_radiance.Load(int3(pre_texel_pos.xy,0));
```

lighting importance sampling pdf visualization:
<p align="center">
    <img src="/resource/simlumen/image/lighting_is_pdf_vis.png" width="60%" height="60%">
</p>

### Structured Importance Sampling

Unreal Lumen adopted a new mechanism called structured importance sampling to reassign the not important samples to those importance directions.

The first step is to calculate the PDF of probe 8x8 pixels. We can obtain the world ray direction by pixel's uv in the group based on the equi area spherical mapping algorithm. The BRDF PDF in this direction can be calculated by the BRDF PDF SH computed in the last pass.

<p align="center">
    <img src="/resource/simlumen/image/struct_id_cull.png" width="70%" height="70%">
</p>


```cpp
// brdf pdf
FThreeBandSHVector brdf;
brdf.V0.x = brdf_pdf_sh[sh_base_idx + 0];
brdf.V0.y = brdf_pdf_sh[sh_base_idx + 1];
brdf.V0.z = brdf_pdf_sh[sh_base_idx + 2];
brdf.V0.w = brdf_pdf_sh[sh_base_idx + 3];
brdf.V1.x = brdf_pdf_sh[sh_base_idx + 4];
brdf.V1.y = brdf_pdf_sh[sh_base_idx + 5];
brdf.V1.z = brdf_pdf_sh[sh_base_idx + 6];
brdf.V1.w = brdf_pdf_sh[sh_base_idx + 7];
brdf.V2.x = brdf_pdf_sh[sh_base_idx + 8];

float2 probe_uv = (group_thread_idx.xy + float2(0.5f,0.5f)) / PROBE_SIZE_2D;
float3 world_cone_direction = EquiAreaSphericalMapping(probe_uv);

FThreeBandSHVector direction_sh = SHBasisFunction3(world_cone_direction);
float pdf = max(DotSH3(brdf, direction_sh), 0);

float light_pdf = light_pdf_tex.Load(int3(dispatch_thread_idx.xy,0));
bool is_pdf_no_culled_by_brdf = pdf >= MIN_PDF_TRACE;

float light_pdf_scaled = light_pdf * PROBE_SIZE_2D * PROBE_SIZE_2D;
pdf *= light_pdf_scaled;
if(is_pdf_no_culled_by_brdf)
{
    pdf = max(pdf, MIN_PDF_TRACE);
}
```

Perform a GPU sort from low to high to find those ray directions that need refinement.

<p align="center">
    <img src="/resource/simlumen/image/struct_is_flow.png" width="50%" height="50%">
</p>

We refine the rays in groups of three. If the maximum PDF among the three rays is less than the minimum PDF threshold, these samples are discarded and refinement is performed. The ray refinement is similar to the mip map. Double the coordinates, compute the local coordinates based on the ray indexes (0,1), (1,1), and (1,0), respectively. Coordinate (0,0) is the position of the corresponding ray to the refine group.

<p align="center">
    <img src="/resource/simlumen/image/struct_is_0.png" width="65%" height="65%">
</p>

```cpp
uint merge_thread_idx = thread_idx % 3;
uint merge_idx = thread_idx / 3;
uint ray_idx_to_refine = max((int)PROBE_SIZE_2D * PROBE_SIZE_2D - (int)merge_idx - 1, 0);
uint ray_idx_to_merge = merge_idx * 3 + 2;

if(ray_idx_to_merge < ray_idx_to_refine)
{
    uint2 ray_tex_coord_to_merge;
	uint ray_level_to_merge;
	float ray_pdf_to_merge;
	UnpackRaySortInfo(RaysToRefine[sort_offset + ray_idx_to_merge], ray_tex_coord_to_merge, ray_level_to_merge, ray_pdf_to_merge);

    if(ray_pdf_to_merge < MIN_PDF_TRACE)
    {
        uint2 origin_ray_tex_coord;
        uint original_ray_level;
        uint original_pdf;
        UnpackRaySortInfo(RaysToRefine[sort_offset + ray_idx_to_refine], origin_ray_tex_coord, original_ray_level, original_pdf);

        RaysToRefine[sort_offset + thread_idx] = PackRaySortInfo(origin_ray_tex_coord * 2 + uint2((merge_thread_idx + 1) % 2, (merge_thread_idx + 1) / 2), original_ray_level - 1, 0.0f);

		if (merge_idx == 0)
		{
			InterlockedAdd(num_rays_to_subdivide, 1);
		}
    }
}
```

## Screen Space Probe Trace

Unreal Lumen is a hybird global illumination solution consisting four GI methods: SSGI, Mesh SDF trace, global SDF trace and cube map. Unreal Lumen performs screen space GI first by sampling the previous scene color. When the SSGI fails to hit, Unreal Lumen performs a mesh SDF trace and samples the surface cache final lighting atlas. Mesh SDF trace is only performed on probes near the camera (positions in 40m radius around the camera). For those further probes, Unreal Lumen performs Global SDF trace and samples the scene voxel lighting. If all of these methods fail, Unreal Lumen falls back to sample the cube map.

In SimLumen, we only perform **two trace methods**: sampling the **surface cache final lighting** by mesh SDF trace and sampling the **scene voxel lighting** by global SDF trace. 

<p align="center">
    <img src="/resource/simlumen/image/hybrid_gi.png" width="65%" height="65%">
</p>

### Screen Space Probe Mesh SDF Trace

We perform mesh SDF trace if the probe's distance to the camera is less than 100m.

Sample 64 directions for each screen space probe. The sample direction is obtained from structured importance sampling table. This table stores the ray coordinates. The refined ray's coordinates (mip 0) range from (0,0) to (16,16), and the other ray's coordinates (mip 1) range from (0,0) to (8,8). Divide the ray coordinates by the mip size and transform them into range (0,1). We can obtain the mapped direction by EquiAreaSphericalMapping with mapped UV.

```cpp
void GetScreenProbeTexelRay(uint2 buffer_idx, inout float3 ray_direction)
{
    uint packed_ray_info = structed_is_indirect_table.Load(int3(buffer_idx.xy,0));

    uint2 texel_coord;
    uint level;
    UnpackRayInfo(packed_ray_info, texel_coord, level);

    uint mip_size = 16 >> level;
    float inv_mip_size = 1.0f / float(mip_size);

    float2 probe_uv = (texel_coord + float2(0.5, 0.5)) * inv_mip_size;
    ray_direction = EquiAreaSphericalMapping(probe_uv);
    ray_direction = normalize(ray_direction);
}
```

Then trace the scene mesh SDFs that are around the current probe in the world space.

```cpp
[loop]
for(uint mesh_idx = 0; mesh_idx < SCENE_SDF_NUM; mesh_idx++)
{
    SMeshSDFInfo mesh_sdf_info = scene_sdf_infos[mesh_idx];
    float trace_bais = mesh_sdf_info.volume_brick_size * 0.125 * 2.0; //2 voxel
    RayTraceSingleMeshSDF(probe_world_position + world_normal * trace_bais, ray_direction, 1000, mesh_idx, trace_result);
}
```

Calculate the hit world position if the ray hits the scene SDFs. Then, sample the surface cache card based on the world position. Unreal Lumen samples the surface cache cards **three times** based on it's ray directions and the normal in the hit position. Considering our meshes are not too complex, SimLumen only **samples once** based on the **maximum ray direction**.

The surface cache card index is calculated from the mesh card start index and the card direction offset index.

With the mesh card index, we can get the surface cache atlas UV and sample the final lighting.

```cpp
float3 hit_world_position = float3(0.0,0.0,0.0);
if(trace_result.is_hit)
{
    SMeshSDFInfo mesh_sdf_info = scene_sdf_infos[trace_result.hit_mesh_index];

    float trace_bais = mesh_sdf_info.volume_brick_size * 0.125 * 2.0; //2 voxel
    hit_world_position = probe_world_position + world_normal * trace_bais + ray_direction * trace_result.hit_distance;
    
    float max_ray_dir = 0.0;
    int max_dir_card_idx = 0;
    if(abs(ray_direction.x) > max_ray_dir)
    {
        if(ray_direction.x > 0){max_dir_card_idx = 5;}
        else{max_dir_card_idx = 4;};
    }
    if(abs(ray_direction.y) > max_ray_dir)
    {
        if(ray_direction.y > 0){max_dir_card_idx = 3;}
        else{max_dir_card_idx = 2;};
    }
    if(abs(ray_direction.z) > max_ray_dir)
    {
        if(ray_direction.z > 0){max_dir_card_idx = 1;}
        else{max_dir_card_idx = 0;};
    }

    uint card_index = mesh_sdf_info.mesh_card_start_index + max_dir_card_idx;
    
    SCardInfo card_info = scene_card_infos[card_index];
    float2 card_uv = GetCardUVFromWorldPos(card_info, hit_world_position);
    
    uint2 card_index_xy = uint2((card_index % card_num_xy), (card_index / card_num_xy));
    uint2 pixel_pos = card_index_xy * 128 + float2(card_uv * 128);
    pixel_pos.y = SURFACE_CACHE_TEX_SIZE - pixel_pos.y;
    
    float3 lighting = surface_cache_final_lighting.Load(int3(pixel_pos,0)).xyz;
    trace_radiance = lighting;
}
else
{
    trace_radiance = float3(0.1,0.1,0.1); //hack sky light
}
```

surface cache sample result:

<p align="center">
    <img src="/resource/simlumen/image/mesh_sdf_trace_result.png" width="85%" height="85%">
</p>

### Screen Space Probe Voxel Trace

The probes further than 100m using voxel lighting trace, which is similar to surface cache sampling. There are several differences between them: the probes trace the scene using a low resolution global SDF rather than single mesh SDFs, and they sample voxel lighting rather than surface cache lighting.

```cpp
SGloablSDFHitResult hit_result = (SGloablSDFHitResult)0;
TraceGlobalSDF(probe_world_position + world_normal * gloabl_sdf_voxel_size * 2.0, ray_direction, hit_result);
```

Voxel lighting is sampled three times based on the ray trace direction. The sample weights are computed by the dot product result between the ray trace direction and voxel face direction.

```cpp
int x_dir = 0;
if(ray_direction.x > 0.0) { x_dir = 5;  }
else { x_dir = 4;  }

int y_dir = 0;
if(ray_direction.y > 0.0) { y_dir = 3;  }
else { y_dir = 2;  }

int z_dir = 0;
if(ray_direction.z > 0.0) { z_dir = 1;  }
else { z_dir = 0;  }


if(hit_result.bHit)
{
    float3 hit_world_position =  ray_direction * (hit_result.hit_distance - gloabl_sdf_voxel_size) + probe_world_position + world_normal * gloabl_sdf_voxel_size * 2.0;

    uint voxel_index_1d = GetVoxelIndexFromWorldPos(hit_world_position);
    SVoxelLighting voxel_lighting = scene_voxel_lighting[voxel_index_1d];

    float3 voxel_lighting_x = voxel_lighting.final_lighting[x_dir];
    float3 voxel_lighting_y = voxel_lighting.final_lighting[y_dir];
    float3 voxel_lighting_z = voxel_lighting.final_lighting[z_dir];

    float weight_x = saturate(dot(ray_direction, voxel_light_direction[x_dir]));
    float weight_y = saturate(dot(ray_direction, voxel_light_direction[y_dir]));
    float weight_z = saturate(dot(ray_direction, voxel_light_direction[z_dir]));

    radiance += (voxel_lighting_x * weight_x);
    radiance += (voxel_lighting_y * weight_y);
    radiance += (voxel_lighting_z * weight_z);

    radiance /= (weight_x + weight_y + weight_z);
}
else
{
    radiance = float3(0.1,0.1,0.1); //hack sky light
}
```

voxel lighting sampling result:

<p align="center">
    <img src="/resource/simlumen/image/voxel_trace_result.png" width="85%" height="85%">
</p>


## Traced Radiance Composite

Since the samples may be refined, we perform an additional composition pass to accumulate the ray samples. Each comptue group processes one probe ( 8x8 samples).

```cpp
groupshared uint shared_accumulator[PROBE_SIZE_2D * PROBE_SIZE_2D][3];
```
The refined ray weighs 1/4 and the other ray weighs 1.

```cpp
const float sample_weight = (float)8 / mip_size * 8 / mip_size; // level 0: weight 1/4, level 1: weight 1
float3 lighting = screen_space_trace_radiance.Load(int3(dispatch_thread_idx.xy, 0)).xyz * sample_weight;
```

 We acclumulate the samples in a probe and assign the result to the original unrefined sample position.

```cpp
uint2 remaped_texel_coord = texel_coord * PROBE_SIZE_2D / mip_size;
uint remapped_thread_idx = remaped_texel_coord.y * PROBE_SIZE_2D + remaped_texel_coord.x;

uint3 quantized_lighting = lighting * lighting_quantize_scale;

InterlockedAdd(shared_accumulator[remapped_thread_idx][0], quantized_lighting.x);
InterlockedAdd(shared_accumulator[remapped_thread_idx][1], quantized_lighting.y);
InterlockedAdd(shared_accumulator[remapped_thread_idx][2], quantized_lighting.z);
```
```cpp
uint thread_index = group_thread_idx.y * PROBE_SIZE_2D + group_thread_idx.x;
radiance = float3(shared_accumulator[thread_index][0], shared_accumulator[thread_index][1], shared_accumulator[thread_index][2]) / lighting_quantize_scale;
```
<p align="center">
    <img src="/resource/simlumen/image/composition.png" width="75%" height="75%">
</p>

radiance composition result:

<p align="center">
    <img src="/resource/simlumen/image/composited_result.png" width="85%" height="85%">
</p>

## Screen Radiance Filter

Filter the screen space radiance to denoise the result. Radiance filter weight is a combination of **angle weight and depth weight** in Unreal Lumen. We use a **simple uniform weight** instead of an angle/depth weight because our scene is simple and most of the mesh is cubes.

<p align="center">
    <img src="/resource/simlumen/image/radiance_filter.png" width="40%" height="40%">
</p>

```cpp
        float3 total_radiance = screen_space_radiance.Load(int3(dispatch_thread_idx.xy, 0)).xyz;
        float total_weight = 1.0;

        int2 offsets[4]; offsets[0] = int2(-1, 0); offsets[1] = int2(1, 0); offsets[2] = int2(0, -1); offsets[3] = int2(0, 1);

        for (uint offset_index = 0; offset_index < 4; offset_index++)
        {
            int2 neighbor_index = offsets[offset_index] * PROBE_SIZE_2D + dispatch_thread_idx.xy;
            if((neighbor_index.x >= 0) && (neighbor_index.x < is_pdf_thread_size_x) && (neighbor_index.y >= 0) && (neighbor_index.y < is_pdf_thread_size_y))
            {
                float neigh_depth = gbuffer_depth.Load(int3(ss_probe_atlas_pos.xy + offsets[offset_index] * PROBE_SIZE_2D, 0));
                if(neigh_depth != 0)
                {
                    total_radiance += screen_space_radiance.Load(int3(dispatch_thread_idx.xy, 0)).xyz;
                    total_weight += 1.0;
                }
            }
        }

        filtered_radiance = total_radiance / total_weight;
```

filtered radiance result:
<p align="center">
    <img src="/resource/simlumen/image/filtered_radiance.png" width="80%" height="80%">
</p>


## Convert To Oct SH Representaion

To denoise the radiance result further, we convert the radiance into **shperical harmonic**, which acts as a low-pass filter. After that we transform the SH into **octahedron** representation. The reason we perform this additional pass is to use **hardware bilinear filter** in the next integration pass. In order to avoid sampling the other probe's result, we add a pixel **board** around the center.

<p align="center">
    <img src="/resource/simlumen/image/radiance_to_oct.png" width="35%" height="35%">
</p>

```cpp
uint2 texel_coord_with_boarder = OctahedralMapWrapBorder(uint2(write_idx_x, write_idx_y),SCREEN_SPACE_PROBE,1);
float2 probe_texel_center = float2(0.5, 0.5);
float2 probe_uv = (texel_coord_with_boarder + probe_texel_center) / (float)SCREEN_SPACE_PROBE;
float3 texel_direction = EquiAreaSphericalMapping(probe_uv);

FThreeBandSHVector diffuse_transfer = CalcDiffuseTransferSH3(texel_direction, 1.0f);
float3 irradiance = 4.0f * PI * DotSH3(irradiance_sh, diffuse_transfer);
screen_space_oct_irradiance[texel_screen_pos] = irradiance;
```
octahedron spherical  harmonic result:

<p align="center">
    <img src="/resource/simlumen/image/oct_result.png" width="70%" height="70%">
</p>

## Integrate

With the boardered octahedron spherical harmonic, we can sample the sh by **hardware bilinear filter**, which denoises the result further.

```cpp
float3 GetScreenProbeIrradiance(uint2 probe_start_pos, float2 irradiance_probe_uv)
{
    float2 sub_pos = irradiance_probe_uv * (SCREEN_SPACE_PROBE - 1.0) + 1.0;
    float2 texel_uv = (probe_start_pos * SCREEN_SPACE_PROBE + sub_pos) / float2(is_pdf_thread_size_x, is_pdf_thread_size_y);
    return screen_space_oct_irradiance.SampleLevel(sampler_linear_clamp,texel_uv,0);
}
```

For each screen pixel, we sample 5 screen probes and accumulate the samples. After dividing the result by the sum of the weights, we are able to obtain the final result for screen indirect lighting:

<p align="center">
    <img src="/resource/simlumen/image/screen_indirect_lighting_result.png" width="70%" height="70%">
</p>

# Result

direct lighting only:
<p align="center">
    <img src="/resource/simlumen/image/direct_lighting_only.png" width="70%" height="70%">
</p>

with GI:
<p align="center">
    <img src="/resource/simlumen/image/with_gi.png" width="70%" height="70%">
</p>

[<u>**Simplified Lumen Source Code**</u>](https://github.com/ShawnTSH1229/SimLumen)




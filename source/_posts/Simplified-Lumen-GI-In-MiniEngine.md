---
title: Simplified Lumen GI In MiniEngine
date: 2024-05-18 13:47:44
categories:
- My Projects
---

# Introduction
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

In Unreal, a mesh may have many mesh cards based on mesh complexity. SimLumen simplifies mesh card generation: generates a fixed number mesh card (6 direction) based on the mesh bounding box.  

<p align="center">
    <img src="/resource/simlumen/image/mesh_card.png" width="40%" height="40%">
</p>

We can calculate the capture position and direction from the bounding box directly. The mesh card depth is determined by software raytracing. For each texel in the mesh card, we trace a ray in the mesh card direction, calculate the ray-mesh intersection and find the furthest intersection distance.
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

Surface Cache Direct Lighting Visualization:
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

Voxel lighting visualization:

<p align="center">
    <img src="/resource/simlumen/image/voxel_lighting_visualize.png" width="60%" height="60%">
</p>

## Surface Cache Indirect Lighting

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

Radiance trace result:
<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_atlas.png" width="60%" height="60%">
</p>

Radiance trace visualization:
<p align="center">
    <img src="/resource/simlumen/image/scache_radiosity_vis.png" width="60%" height="60%">
</p>

### Radiosity Filter
使用当前 Radiosity Texel 的 World Normal 和 Neighbor 的 World Position 构建 Plane。
计算当前 Radiosity Texel 到这个平面的距离。
计算相对距离，使用第 2 步的距离除以两个点的距离，最小不小于 0.1。
根据相对距离计算 Depth Weight，这里通过 ProbePlaneWeightingDepthScale 系数调节，默认 -100。
根据 Depth Weight 返回权重，这里只有两种权重：0 和 1，这意味着 Neighbor 的 Radiance 没有中间过渡状态，只有是否影响当前 Radiosity Texel 两种状态，这可以有效防止 Light Leaking 的问题。

### Convert To SH
### Radiosity Integrate

# Final Gather
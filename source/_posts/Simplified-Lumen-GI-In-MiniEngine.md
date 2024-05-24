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

We use Embree to calculate the mesh SDF, which is the same as Unreal does. For each voxel, SimLumen traces 512 ray samples to intersect with the mesh and finds the closest hit position between these samples. A voxel is considered within the mesh if more than 25% of the 512 ray samples hit the triangle backface. We take the negative value of the closest hit distance.

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

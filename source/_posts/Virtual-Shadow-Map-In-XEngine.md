---
title: Virtual Shadow Map In XEngine
date: 2024-05-01 18:51:18
tags:
categories:
- My Projects
---
# Introduction

Virtual Shadow Maps (VSMs) is the new shadow mapping method used in Unreal Engine 5. I implemented a **simplified virtual shadow maps** in my **personal game engine**. Here is a brief introduction from the official unreal engine 5 documentation:
>Virtual Shadow Maps have been developed with the following goals:
>
>* Significantly increase shadow resolution to match highly detailed Nanite geometry
>* Plausible soft shadows with reasonable, controllable performance costs
>* Provide a simple solution that works by default with limited amounts of adjustment needed
>* Replace the many Stationary Light shadowing techniques with a single, unified path
>
>Conceptually, virtual shadow maps are just very **high-resolution** shadow maps. In their current implementation, they have a **virtual resolution** of 16k x 16k pixels. **Clipmaps** are used to increase resolution further for Directional Lights. To keep performance high at reasonable memory cost, VSMs split the >shadow map into tiles (or Pages) that are 128x128 each. Pages are allocated and rendered only as needed to shade **on-screen pixels** based on an analysis of the depth buffer. The pages are **cached** between frames unless they are invalidated by moving objects or light, which further improves performance.

According to the Unreal Engine VSMs documentation, VSMs have four key features: virtual high-resolution texture, clipmaps, only shade on-screen pixels and page cache. We have implemented the four features listed above in the simplified virtual shadow map project. Here is the low-level architecture of our simplified VSMs:

<p align="center">
    <img src="/resource/vsm_project/image/low_level_architecture.png" width="70%" height="70%">
</p>

In XEngine's simplified virtual shadow map, each directional light has a clip map with 3 levels: clip map level 6, clip map level 7 and clip map level 8. In addition, the maximum clip map level consists of 8 x 8 tiles, with the next clip map level having twice the number of tiles as the previous clip map level. The total tile number is 1344 (32 x 32 + 16 x 16 + 8 x 8), which corresponds to 8K x 8K virtual texture size. Each tile have the same physical size with 128 x 128 pixels. The physical tile pool's size is 4K x 4K.
# Pass Overview

# Mark Tile State

Only the tiles that affect objects in the camera view are processed by VSMs. In order to achieve this goal, VSMs analyze the scene depth buffer by convert the pixels from camera projection space to shadow view space to find which tile this pixel corresponds to. Moreover, VSMs calculate the clip map level this pixel belongs to based on its distance from world space to the camera. As shown in the following image, the green blocks are marked as used tiles in VSMs, while the red blocks are not used.

<p align="center">
    <img src="/resource/vsm_project/image/vsm_tile_mark_drawio.png" width="50%" height="50%">
</p>

Simplified VSMs have 3 clip map levels. The ranges of each mip level are 2^(6 + 1) ,2^(7 + 1)  and 2^(8 + 1). The first level is 6, which means that the pixels distance to the camera ranging from 0 to 2^6 are belong to this level. Based on the mip level, shadow space UV, and mip size, we can determine which tile this pixel belongs to.

```cpp
float Distance = length(WorldCameraPosition - WorldPosition.xyz);
float Log2Distance = log2(Distance + 1);
 
int MipLevel = clamp(Log2Distance - VSM_CLIPMAP_MIN_LEVEL, 0, VSM_MIP_NUM - 1);
uint2 VSMTileIndex = uint2(UVOut * MipLevelSize[MipLevel]);

int DestTileInfoIndex = MipLevelOffset[MipLevel] + VSMTileIndex.y * MipLevelSize[MipLevel] + VSMTileIndex.x;

VirtualShadowMapTileState[DestTileInfoIndex] = TILE_STATE_USED;
```
Here is a visualization of the mip level for the scene. The Mip Levels 6 / 7 / 8 correspond to the colors red, green, and blue, respectively. 

<p align="center">
    <img src="/resource/vsm_project/image/vsm_tile_mark_visualize.png" width="80%" height="80%">
</p>

# Mark Cache Miss Tile

# Update Tile State

# Allocate The Physical Tile

# Cull And Build The Shadow Draw Command

# Shadow Map Rendering

# Generate The Shadow Mask Texture


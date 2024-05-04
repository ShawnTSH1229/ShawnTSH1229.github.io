---
title: Virtual Shadow Map In XEngine
date: 2024-05-01 18:51:18
tags:
index_img: /resource/vsm_project/image/index_image.png
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

In XEngine's simplified virtual shadow map, each directional light has a clip map with 3 levels: clip map level 6, clip map level 7 and clip map level 8. In addition, the maximum clip map level consists of 8 x 8 tiles, with the next clip map level having twice the number of tiles as the previous clip map level. The total tile number is 1344 (32 x 32 + 16 x 16 + 8 x 8), which corresponds to 8K x 8K virtual texture size. Each tile have the same physical size with 256 x 256 pixels. The physical tile pool's size is 2K x 2K.
# Overview
Note: To achieve the **highest possible performance**, it is recommended that combining the virtual shadow map with a **Nanite-like** technique which divides the mesh into clusters. Otherwise, **fine-grained tile splitting** could result in significant increases in the draw call and the mesh face number.

<p align="center">
    <img src="/resource/vsm_project/image/renderdoc.png" width="64%" height="64%">
</p>

Simplified VSMs classify the tiles into different states based on the scene depth and the mesh bound box. It contains four tile states: shaded, not visible, cache miss and newly added tile state. VSMs performs different action for each tile with its tile state, such as update the physical tile content, allocate or remove the tile in the physical tile pool. Next, cull and build the mesh draw command for those tiles that need to be updated in the current frame. Finally, dispatch a set of indirect draw commands and compute the shadow mask using the shadow depth map rendered by indirect draw:

<p align="center">
    <img src="/resource/vsm_project/image/pass_overview.png" width="64%" height="64%">
</p>

# Mark Tile State

## Mark Tiles Used

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

## Mark Cache Miss Tiles

VSMs only update the tiles that have changed compared to previous frame. This reduces the number of draw calls since we can reuse the previous frame data in the cached virtual shadow map texture. To find all of the tiles that changed in this frame, simplified VSMs project the bound boxes of the dynamic objects into the shadow view space. Following that, the VSM iterates and marks all tiles within these bound boxes projected range. The VSMs only mark tiles as cache misses for **those tiles that were rendered in this frame**. All of the mip levels covered are conservatively marked as cache miss. In the following gif image, we can see that the cache missed tile with yellow color is updated every frame:

<p align="center">
    <img src="/resource/vsm_project/gif/tile_cache_miss.gif" width="80%" height="80%">
</p>

# Update Tile Action

In the tile action update pass, VSMs compare the current tile state with the previous frame's tile state. A ping-pong buffer is used to store the previous's tile state buffer and tile table buffer. The tile table buffer stores the index to the physical tile texture.

<p align="center">
    <img src="/resource/vsm_project/image/update_tile_action.png" width="50%" height="50%">
</p>

Newly added tiles in this frame will allocate a new physical tile and update the shadow rendering. For cached tiles, we only update the tile shadow rendering. There is no need to allocate an extra physical tile for the cached tile. We reuse the cached tile by assign it with the index copyed from the corresponding position in the previoues tile table.
```cpp
    bool bTileUsed = (TileState == TILE_STATE_USED);
    bool bPreTileUsed = (PreTileState == TILE_STATE_USED);
    bool bNewTile = (TileState == TILE_STATE_USED) && (PreTileState == TILE_STATE_UNUSED);
    bool bCacheMissTile = (CacheMissAction == TILE_STATE_CACHE_MISS) && (PreTileState == TILE_STATE_USED);
    bool bTileRemove = (TileState == TILE_STATE_UNUSED) && (PreTileState == TILE_STATE_USED);
    bool bTileActionCached = (bPreTileUsed) && (bTileUsed) && (!bCacheMissTile);
    bool bTileActionNeedRemove = bTileRemove;

    uint TileAction = 0;

    if(bTileActionCached)
    {
        VirtualShadowMapTileTable[GlobalTileIndex] = VirtualShadowMapPreTileTable[GlobalTileIndex];
        TileAction = TILE_ACTION_CACHED;
    }

    if(bCacheMissTile)
    {
        VirtualShadowMapTileTable[GlobalTileIndex] = VirtualShadowMapPreTileTable[GlobalTileIndex];
        TileAction = TILE_ACTION_NEED_UPDATE;       
    }
    
    if(bNewTile)
    {
        TileAction = TILE_ACTION_NEED_ALLOCATE | TILE_ACTION_NEED_UPDATE;
    }

    if(bTileActionNeedRemove)
    {
        TileAction = TILE_ACTION_NEED_REMOVE;
    }

    VirtualShadowMapTileAction[GlobalTileIndex] = TileAction;
```
# Physical Tile Management

VSMs maintain a list of available physical tiles. An extra counter buffer records the free list header node. The physical tile manager allocates a free tile from the free tile list when the tile action equals TILE_ACTION_NEED_ALLOCATE and assigns the tile index to the corresponding position of the current frame virtual tile table when the tile action equals TILE_ACTION_NEED_ALLOCATE. The physical tile manager will obtain the released tile index from the previous frame tile table and push it to the back of the free tile list if the tile action is TILE_ACTION_NEED_REMOVE.

<p align="center">
    <img src="/resource/vsm_project/image/physical_tile_manage.png" width="80%" height="80%">
</p>

We move the counter forward or backward by InterlockedAdd instruction. The tile realse and tile allocate action are performed in separate compute pass. VSMs in UE5 maintain a LRU list in the physical tile manager. For simplicity, we release the tile buffer immediately after the tiles are marked as TILE_ACTION_NEED_REMOVE.

```cpp
    if(VirtualShadowMapAction & TILE_ACTION_NEED_ALLOCATE)
    {
        uint FreeListIndex = 0;
        InterlockedAdd(VirtualShadowMapFreeListStart[0],1,FreeListIndex);
        if(FreeListIndex < VSM_TEX_PHYSICAL_WH * VSM_TEX_PHYSICAL_WH)
        {
            VirtualShadowMapTileTable[GlobalTileIndex] = VirtualShadowMapFreeTileList[FreeListIndex];
        }
    }
```

```cpp
    if(VirtualShadowMapAction & TILE_ACTION_NEED_REMOVE)
    {
        int FreeListIndex = 0;
        InterlockedAdd(VirtualShadowMapFreeListStart[0],-1,FreeListIndex);
        if(FreeListIndex > 0)
        {
            uint RemoveTileIndex = VirtualShadowMapPreTileTable[GlobalTileIndex];
            VirtualShadowMapFreeTileList[FreeListIndex - 1] = RemoveTileIndex;
        }
    }
```
When rotating the camera, we can see that the free tiles (red blocks in the top-right corner) are changed:

<p align="center">
    <img src="/resource/vsm_project/gif/physical_tile_allocation.gif" width="64%" height="64%">
</p>

# Cull And Build The Shadow Draw Command
## Allocate The Shadow Commands

VSMs build the command on the GPU and render the shadow map by indirect draw command. Here is the indirect command layout specified on the application side: constant buffer view for per-object transform information, tile info cbv, vertex buffer view, index buffer view and indirect argument desc.

<p align="center">
    <img src="/resource/vsm_project/image/indirect_cmd_layout.png" width="50%" height="50%">
</p>

After that, we allocate the command data on the application side. It contains the GPU address of the buffer, information about the vertex buffer and index buffer, as well as information about the object mesh. In the final step, initialize the scene command buffer with the data allocated above and create an empty culled command buffer with the same size as the scene command buffer. The culled command buffer is a collection of commands used in shadow map rendering after GPU culling.

```cpp
std::vector<XRHICommandData> RHICmdData;
RHICmdData.resize(RenderGeos.size());
for (int i = 0; i < RenderGeos.size(); i++)
{
	auto& it = RenderGeos[i];
	RHICmdData[i].CBVs.push_back(it->GetAndUpdatePerObjectVertexCBuffer().get());
	RHICmdData[i].CBVs.push_back(VirtualShadowMapResource.LightSubProjectMatrix.get());
	RHICmdData[i].CBVs.push_back(VirtualShadowMapResource.LightSubProjectMatrix.get());

	auto VertexBufferPtr = it->GetRHIVertexBuffer();
	auto IndexBufferPtr = it->GetRHIIndexBuffer();
	RHICmdData[i].VB = VertexBufferPtr.get();
	RHICmdData[i].IB = IndexBufferPtr.get();
	RHICmdData[i].IndexCountPerInstance = it->GetIndexCount();
	RHICmdData[i].InstanceCount = 1;
	RHICmdData[i].StartIndexLocation = 0;
	RHICmdData[i].BaseVertexLocation = 0;
	RHICmdData[i].StartInstanceLocation = 0;
}

uint32 OutCmdDataSize;
void* DataPtrret = RHIGetCommandDataPtr(RHICmdData, OutCmdDataSize);
```
## Build Indirect Shadow Command

For each tile, we dispatch 50 threads to process the mesh batch. Each mesh batch has 1 / 50 mesh draw commands. Simplified VSMs cull the mesh draw command from the mesh bounding box. We project the box into the shadow view space, and push the command to the output command queue by InterLockedAdd when the mesh is not culled by the tile.

<p align="center">
    <img src="/resource/vsm_project/image/build_shadow_command.png" width="75%" height="75%">
</p>

## GPU "Pointer"

It is necessary to create a bridge between the indirect draw command and the virtual tile table in order to obtain tile information. UE5's VSMs use InstanceID to index the virtual tile table. The solution requires recording an extra InstanceOffset variable for each mesh, since the StartInstanceLocation cannot be obtained from the vertex shader if the shading language is beyond SM 6.8. In Simplified VSMs, we use **"GPU Pointer"** to point the GPU address of the tile information buffer.

<p align="center">
    <img src="/resource/vsm_project/image/gpu_pointer.png" width="75%" height="75%">
</p>

For each tile, simplified VSMs generate information including a view-proj matrix, mip level, and tile indexes. We can calculate the GPU address of each tile with the tile index and the base GPU address recorded on the application side.

Another problem is that the GPU pointer is 64 bits in size, and HLSL does not support additions of 64 bits. Therefore, it is necessary to implement a custom 64 bit addition for the GPU pointer:
```cpp
struct UINT64
{
    uint LowAddress;
    uint HighAddress;
};

UINT64 UINT64_ADD(UINT64 InValue , uint InAdd)
{
    UINT64 Ret = InValue;
    uint C= InValue.LowAddress + InAdd;
    bool OverFlow = (C < InValue.HighAddress) || (C < InAdd);
    if(OverFlow)
    {
        Ret.HighAddress += 1;
    }
    Ret.LowAddress = C;
    return Ret;
}
```
A tile's GPU address is simply the sum of the tile offset and the base address.
```cpp
uint2 TileIndexMin = uint2( UVMin * MipLevelSize[MipLevel]);
uint2 TileIndexMax = uint2( UVMax * MipLevelSize[MipLevel]);

if(TileIndexMin.x <= MipTileIndexXY.x && TileIndexMin.y <= MipTileIndexXY.y && TileIndexMax.x >= MipTileIndexXY.x && TileIndexMax.y >= MipTileIndexXY.y)
{
    uint PointerOffset = GlobalTileIndex * ( 4 * 4 * 4 + 4 * 4); // float4x4 + uint4
    ShadowIndirectCommand InputCommand = InputCommands[Index];
    InputCommand.CbGlobalShadowViewProjectAddressVS = UINT64_ADD(GPUStartAddress,PointerOffset);
    InputCommand.CbGlobalShadowViewProjectAddressPS = InputCommand.CbGlobalShadowViewProjectAddressVS;
    InputCommand.StartInstanceLocation = GlobalTileIndex;
    
    uint OriginalValue = 0;
    InterlockedAdd(CommandCounterBuffer[0], 1, OriginalValue);
    OutputCommands[OriginalValue] = InputCommand;
}
```

# Shadow Map Rendering
After all pre-requirements have been met, shadow map rendering is an easy task-just clear the physical tiles that need to be updated and dispatch an indirect draw pass.

```cpp
RHICmdList.RHIExecuteIndirect(VirtualShadowMapResource.RHIShadowCommandSignature.get(), RenderGeos.size() * 16,
	VirtualShadowMapResource.VirtualShadowMapCommnadBufferCulled.GetBuffer().get(), 0,
	VirtualShadowMapResource.VirtualShadowMapCommnadCounter.GetBuffer().get(), 0);
```
By calculating the MIP level based on the world position and calculating the virtual table index based on the GPU pointer, we can determine destination pixel writen position.

```cpp
uint TableIndex = MipLevelOffset[MipLevel] + VirtualTableIndexY * MipLevelSize[MipLevel] + VirtualTableIndexX;
uint TileInfo = VirtualShadowMapTileTable[TableIndex];
uint PhysicalTileIndexX = (TileInfo >>  0) & 0xFFFF;
uint PhysicalTileIndexY = (TileInfo >> 16) & 0xFFFF;

float2 IndexXY = uint2(PhysicalTileIndexX, PhysicalTileIndexY) * VSM_TILE_TEX_PHYSICAL_SIZE;
float2 WritePos = IndexXY + PositionIn.xy;
float FixedPointDepth = float(PositionIn.z) * uint(0xFFFFFFFF);
uint UintDepth = FixedPointDepth;

InterlockedMax(PhysicalShadowDepthTexture[uint2(WritePos)],UintDepth);
```
Here is the visualization of the physical shadow depth texture:
<p align="center">
    <img src="/resource/vsm_project/image/sub_tile.png" width="64%" height="64%">
</p>

<p align="center">
    <img src="/resource/vsm_project/image/physical_tile_visualize.png" width="64%" height="64%">
</p>

# Generate The Shadow Mask Texture

By calculating the virtual table index from the pixel position, calculating the physical shadow depth position from the table index, and comparing the result to the pixel's shadow space depth, we can finally determine whether this pixel is shadowed.

```cpp
float Total = 0.0f;
for(int i = 0; i < 5; i++)
{
    Total += ComputeShadowFactor(UVShadowSpace + UVOffset[i] * MipLevelVirtualTextureSize[MipLevel], ObjectShadowDepth,Bias, MipLevel);
}
```
```cpp
float ComputeShadowFactor(float2 UVShadowSpace ,uint ObjectShadowDepth, uint Bias, uint MipLevel)
{
    uint2 VSMTileIndex = uint2(UVShadowSpace * MipLevelSize[MipLevel]);
    int TableIndex = MipLevelOffset[MipLevel] + VSMTileIndex.y * MipLevelSize[MipLevel] + VSMTileIndex.x;
    uint TileInfo = VirtualShadowMapTileTable[TableIndex];
    uint PhysicalTileIndexX = (TileInfo >>  0) & 0xFFFF;
    uint PhysicalTileIndexY = (TileInfo >> 16) & 0xFFFF;

    float2 IndexXY = uint2(PhysicalTileIndexX, PhysicalTileIndexY) * VSM_TILE_TEX_PHYSICAL_SIZE;
    float2 SubTileIndex = (float2(UVShadowSpace * MipLevelSize[MipLevel]) - VSMTileIndex) * VSM_TILE_TEX_PHYSICAL_SIZE;
    float2 ReadPos = IndexXY + SubTileIndex;

    uint ShadowDepth = PhysicalShadowDepthTexture[uint2(ReadPos)].x;

    if((ObjectShadowDepth + Bias ) < ShadowDepth) 
    {
        return 1.0f;
    }

    return 0.0;
}
```

<p align="center">
    <img src="/resource/vsm_project/image/scene_depth.png" width="64%" height="64%">
</p>

<p align="center">
    <img src="/resource/vsm_project/image/shadow_mask.png" width="64%" height="64%">
</p>

<p align="center">
    <img src="/resource/vsm_project/gif/final_result_visualize.gif" width="64%" height="64%">
</p>


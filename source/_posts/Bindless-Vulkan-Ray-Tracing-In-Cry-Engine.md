---
title: Bindless Vulkan Ray Tracing In Cry Engine
date: 2024-06-30 21:32:51
tags:
index_img: /resource/vkraytracing/image/redraid.png
---

# Acceleration Structure Construction
Both DXR and Vulkan RayTracing use bounding volume hierarchies (BVH) as acceleration structure, which is also commonly used for offline ray tracing. BVH is an object partition data structure. It has more advantages than space partition (BSP-Tree,Octree,etc). One advantage of BVH is that the scene can be rebuilt easily for dynamic objects: just recompute the leaf node based on the current geometry. Additionally, the two-level acceleration structure employed in the current raytracing API can optimize rigid body animation further: only update the transform matrix without the expensive bottom-level acceleration structure update. Another advantage of BVH is that the maximum size used is known, as the number of leaves is limited

## Two-Level Acceleration Structure
Vulkan uses a two-level acceleration structure to accelerate scene transversal and ray intersections. The bottom-level acceleration structure (BLAS) contains a set of geometries. In the Vulkan API, we can set two types of geometries: triangles or procedural. The former type contains a set of triangles, that is, actual vertex data and index data. Additionally, the BLAS with triangle geometry type contains AABB to encapsulate the geometry after it has been built. For the later geometry type, we should specify AABBs and the associated intersection function such as the ray-sphere test function. In practice, we prefer using triangle geometry, since both triangles and AABBs are hardware accelerated, but the procedural type involves additional user defined intersection functions in a shader, which is slower

The top-level acceleration structure (TLAS) consists of instances that reference the BLAS. And each instance contains a transform. We can only update the instance transform of TLAS for rigid-body animation
## Top-level Acceleration Structure Construction

### The Trade-Offs of Acceleration Structure Building
The driver is responsible for scheduling construction tasks after requesting the acceleration structure building. And the highly parallel work (calculate AABB, sort, etc) can accelerated by GPU. In addition, AS management (build/updata) can be moved to an async compute queue, which could completely hide the cost in many cases. AS management is essentialy software-based work, which allows the driver to optimize its construction algorithm continuously.

Vulkan exposed three build options to AS management:
For the PREFER_FAST_TRACE bit, the driver should choose the algorithm with better runtime transversal efficiency.
For the PREFER_FAST_BUILD bit, the driver should choose the algorithm with a faster construction rate.
For the default option, driver should choose a compromise solution with a balance between trace and build speed.
Generally, we employ PREFER_FAST_BUILD for BLAS, as well as PREFER_FAST_TRACE for TLAS.

Ray tracing has more geometries in GPU memory than traditional rasterization methods. Vulkan has provided additional options for compacting the acceleration structure. Generally, we only compact BLAS, since it has more geometry data than TLAS

What is the algorithm implemented in the driver behind the AS build options? There are three algorithms corresponding to PREFER_FAST_TRACE, PREFER_FAST_BUILD and the default option.

The first algorithm is called LBVH (Linear BVH). It uses linear ordering derived from spatial Morton codes to build hierarchies extremely quickly and with high parallel scalability.
<p align="center">
    <img src="/resource/vkraytracing/image/LBVH.png" width="50%" height="50%">
</p>

LBVH is a simple split method and can't guarantee the BVH build quality. SAH is a more commonly used algorithm. It's a heuristic algorithm. It splits the scene meshes based on surface area. However, SAH needs to iterate all scene meshes during construction and take a lot of time during BVH construction.

Binning-SAH is an optimized version of the SAH algorithm. The idea is to divide the node bounding box equally in a certain dimension K (such as 32), and then take the (K-1) equal points as the segmentation boundary (divide the triangles on both sides into two child nodes) to calculate (K-1) costs, and take the segment with the smallest cost. When splitting BVH nodes in complex scenarios, K is much smaller than N. Of course, the final partition might just be a suboptimal solution.

<p align="center">
    <img src="/resource/vkraytracing/image/Bin-SAH.png" width="50%" height="50%">
</p>

When there are unevenly sized triangles in the scene, the bounding boxes of the two child nodes after Binning SAH segmentation may overlap. This causes it to cost across both child nodes.

Spatial-split SAH is an algorithm to further improve the quality of binning SAH by eliminating the overlap problem, the idea being to allow the same triangle to enter two child nodes. For each node, Binning SAH is first used to find the optimal segmentation. If there is no overlap of bounding boxes between the two child nodes separated by Binning SAH, then the segmentation ends. 

<p align="center">
    <img src="/resource/vkraytracing/image/SplitSAH.png" width="50%" height="50%">
</p>

Otherwise, perform spatial-split, and the splitting step is as follows: split the node bounding box into K equal parts in a certain dimension. At (K-1) dividing points, any triangle is divided into the left node as long as some of the bounding box is on the left side of the dividing point. The right node as long as some is on the right side of the dividing point. Calculate the above (K-1) costs and take the partition with the smallest cost. If the cost is smaller than that of Binning SAH, then the Spatial-Split result is adopted.

### AMD TLAS Rebraid

In AMD's implementation, if the TLAS is built without the AllowUpdate flag, AMD GPURT will perform rebraid to improve the TLAS construction quality.

From AMD GPURT source code:
```cpp
    if ((Util::TestAnyFlagSet(m_buildArgs.inputs.flags, AccelStructBuildFlagAllowUpdate) == false) &&
        m_buildConfig.topLevelBuild)
    {
        if (m_buildConfig.rebraidType == RebraidType::V1)
        {
            // inputs > maxTopDownBuildInstances turn off rebraid
            if (m_buildConfig.topDownBuild == false)
            {
                m_buildConfig.rebraidType = RebraidType::Off;
            }
        }
        else
        {
            m_buildConfig.rebraidType = m_buildConfig.rebraidType;
        }
    }
    else
    {
        m_buildConfig.rebraidType = RebraidType::Off;
    }
```

>two objects (green and blue), each with their own BVH; with their topologies (top), and their spatial extents (bottom). As the objects spatially overlap and the top-level BVH (brown) has to treat them as monolithic entities a significant BVH node overlap in the spatial domain occurs, leading to low traversal performance. (b) Our method allows the top-level BVH to look into the object BVHs, and to "open up" object nodes where appropriate. This allows the top-level BVH to create new top-level nodes (brown) that address individual subtrees in the object BVHs, resulting in improved BVH quality
>------from Improved Two-Level BVHs using Partial Re-Braiding

<p align="center">
    <img src="/resource/vkraytracing/image/redraid.png" width="60%" height="60%">
</p>

The core idea of rebraid is to:
1.start with object BVHs in the same way a traditional twolevel BVH would;
2.find a suitable "cut" through each object's BVH such that the resulting set of BVH subtrees has low(er) overlap;
3.build a top-level BVH over those resulting subtrees.

## Bottom-level Acceleration Structure Construction
Bottom-level acceleration structure construction is divided into three steps: construct the create information, create the acceleration structure and build the acceleration structure.
### Construct the create information
In order to create an acceleration structure, we must obtain the size information of the acceleration structure by calling vkGetAccelerationStructureBuildSizesKHR, and then create the acceleration structure buffer with the calculated size. Next, call vkCreateAccelerationStructureKHR with the structure creation information, and store the result in accelerationStructureHandle. It is also necessary to obtain the device address of accelerationStructureHandle. This parameter is required in the next TLAS construction.

It should be noted that a new flag (USAGE_ACCELERATION_STRUCTURE) should be added to the creation of the acceleration structure buffer. It corresponds to XXX_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR in my implementation, which indicates this buffer is suitable for use as a read-only input to an acceleration structure build.

```cpp
m_sSizeInfo = GetDeviceObjectFactory().GetRayTracingBottomLevelASSize(rtBottomLevelCreateInfo);

m_accelerationStructureBuffer.Create(1u, static_cast<uint32>(m_sSizeInfo.m_nAccelerationStructureSize), DXGI_FORMAT_UNKNOWN, CDeviceObjectFactory::USAGE_ACCELERATION_STRUCTURE, nullptr);

VkAccelerationStructureCreateInfoKHR accelerationStructureCreateInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
accelerationStructureCreateInfo.buffer = m_accelerationStructureBuffer.GetDevBuffer()->GetBuffer()->GetHandle();
accelerationStructureCreateInfo.offset = 0;
accelerationStructureCreateInfo.size = m_sSizeInfo.m_nAccelerationStructureSize;
accelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

Extensions::KHR_acceleration_structure::vkCreateAccelerationStructureKHR(m_pDevice->GetVkDevice(), &accelerationStructureCreateInfo, nullptr, &accelerationStructureHandle);

VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
deviceAddressInfo.accelerationStructure = accelerationStructureHandle;
accelerationStructureDeviceAddress = Extensions::KHR_acceleration_structure::vkGetAccelerationStructureDeviceAddressKHR(m_pDevice->GetVkDevice(), &deviceAddressInfo);
```
One of the advantages of BVH over space partitioning is that the structure size is known. Once the required data (vertex/index buffer, format, stride, etc) is ready, we can query the size of bottom-level acceleration structure information. The shape and type of the acceleration structure to be created is described in the VkAccelerationStructureBuildGeometryInfoKHR structure. This is the same structure that will later be used for the actual build, but the acceleration structure parameters and geometry data pointers do not need to be fully populated at this point (although they can be), just the acceleration structure type, and the geometry types, counts, and maximum sizes. These sizes are valid for any sufficiently similar acceleration structure.
```cpp
static void GetBottomLevelAccelerationStructureBuildInfo(
	const VkDevice* pVkDevice,
	const std::vector<SRayTracingGeometryTriangle>& rtGeometryTriangles,
	const SRayTracingBottomLevelASCreateInfo::STriangleIndexInfo& sSTriangleIndexInfo,
	EBuildAccelerationStructureFlag eBuildFlag,
	EBuildAccelerationStructureMode eBuildMode,
	SVulkanRayTracingBLASBuildInfo& outvkRtBLASBuildInfo)
{
	std::vector<uint32> maxPrimitiveCounts;
	for (uint32 index = 0; index < rtGeometryTriangles.size(); index++)
	{
		const SRayTracingGeometryTriangle& rayTracingGeometryTriangle = rtGeometryTriangles[index];
		VkDeviceAddress vertexDeviceAddress = InputStreamGetBufferDeviceAddress(xxxxxx);

		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		......
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		accelerationStructureGeometry.geometry.triangles = xxx;
		outvkRtBLASBuildInfo.m_vkRtGeometryTrianglesInfo.push_back(accelerationStructureGeometry);

		VkAccelerationStructureBuildRangeInfoKHR RangeInfo = {};
		outvkRtBLASBuildInfo.m_vkAsBuildRangeInfo.push_back(RangeInfo);

		maxPrimitiveCounts.push_back(rayTracingGeometryTriangle.m_sTriangVertexleInfo.m_nMaxVertex);
	}

	outvkRtBLASBuildInfo.m_vkAsBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	......
	outvkRtBLASBuildInfo.m_vkAsBuildGeometryInfo.pGeometries = outvkRtBLASBuildInfo.m_vkRtGeometryTrianglesInfo.data();
	Extensions::KHR_acceleration_structure::vkGetAccelerationStructureBuildSizesKHR(*pVkDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &outvkRtBLASBuildInfo.m_vkAsBuildGeometryInfo, maxPrimitiveCounts.data(), &outvkRtBLASBuildInfo.m_vkAsBuildSizeInfo);
}
```
GetBottomLevelAccelerationStructureBuildInfo is a shared function between creating and building BLAS, which is the same as VkAccelerationStructureBuildGeometryInfoKHR. From this function, we can obtain the size for Create BLAS and the build information for Build BLAS.

### Build BLAS

We construct the acceleration structure in batch mode by gathering all of the BLAS of each geometry and composing them into an array.
```cpp
std::vector<CRayTracingBottomLevelAccelerationStructurePtr> rtBottomLevelASPtrs;
for (uint32 index = 0; index < m_nObjectNum; index++)
{
	rtBottomLevelASPtrs.push_back(m_objectGeometry[index]->m_pRtBottomLevelAS);
}
	
pCommandInterface->BuildRayTracingBottomLevelASs(rtBottomLevelASPtrs);
```
In acceleration structure building, scratch buffers are required, but they are not used in ray tracing because acceleration structure buffers require more information, such as AABBs. These extra data are generated in VKCMDBuildAccelerationStructuresKHR. Therefore, the scratch buffer keeps temporary data, which can be released after building.Each BLAS in a batch can share a single scratch buffer, whose size is the total size of all the BLAS.
```cpp
uint64 nTotalScratchBufferSize = 0;
for (auto& blas : rtBottomLevelASPtrs)
{
	nTotalScratchBufferSize += blas->m_sSizeInfo.m_nBuildScratchSize;
}

CGpuBuffer scratchBuffer;
scratchBuffer.Create(1u, static_cast<uint32>(nTotalScratchBufferSize), DXGI_FORMAT_UNKNOWN, CDeviceObjectFactory::USAGE_STRUCTURED | CDeviceObjectFactory::BIND_UNORDERED_ACCESS, nullptr);
```
Lastly, obtain the acceleration structure building geometry information, specify the scratch buffer device address, and build the acceleration structure in batches.
```cpp
std::vector<SVulkanRayTracingBLASBuildInfo> tempBuildInfos;
std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildGeometryInfos;
std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRangeInfos;

uint64 nScratchBufferOffset = 0;
for (auto& blas : rtBottomLevelASPtrs)
{
	tempBuildInfos.push_back(SVulkanRayTracingBLASBuildInfo());
	SVulkanRayTracingBLASBuildInfo& vkRtBLASBuildInfo = tempBuildInfos[tempBuildInfos.size() - 1];

	GetBottomLevelAccelerationStructureBuildInfo(xxxxxx, vkRtBLASBuildInfo);

	VkBufferDeviceAddressInfo bufferDeviceAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
	bufferDeviceAddressInfo.buffer = scratchBuffer.GetDevBuffer()->GetBuffer()->GetHandle();
	VkDeviceAddress scratchBufferAddress = vkGetBufferDeviceAddress(GetDevice()->GetVkDevice(), &bufferDeviceAddressInfo) + nScratchBufferOffset;
	nScratchBufferOffset += blas->m_sSizeInfo.m_nBuildScratchSize;

	CVulkanRayTracingBottomLevelAccelerationStructure* vkRtBLAS = static_cast<CVulkanRayTracingBottomLevelAccelerationStructure*>(blas.get());
	vkRtBLASBuildInfo.m_vkAsBuildGeometryInfo.dstAccelerationStructure = vkRtBLAS->accelerationStructureHandle;
	vkRtBLASBuildInfo.m_vkAsBuildGeometryInfo.srcAccelerationStructure = nullptr;
	vkRtBLASBuildInfo.m_vkAsBuildGeometryInfo.scratchData.deviceAddress = scratchBufferAddress;

	buildGeometryInfos.push_back(vkRtBLASBuildInfo.m_vkAsBuildGeometryInfo);
	buildRangeInfos.push_back(vkRtBLASBuildInfo.m_vkAsBuildRangeInfo.data());
}

VkCommandBuffer cmdBuffer = GetVKCommandList()->GetVkCommandList();
Extensions::KHR_acceleration_structure::vkCmdBuildAccelerationStructuresKHR(cmdBuffer, rtBottomLevelASPtrs.size(), buildGeometryInfos.data(), buildRangeInfos.data());
```
# Ray Tracing Shader Cross Compile
Add vulkan ray tracing shader support for cry engine.

## Add Token
Cry Engine will initialize all shaders in the 'mfInitShadersList' function during Engine initialization. We need to add the tokens related to the ray tracing that are not supported by the Cry Engine parser. In Cry Engine, the token is parsed in the 'm_Bin.GetBinShader->SaveBinShader'  function. To invoke this breakpoint, you need to delete the shader cache located in `user\shaders\cache\vulkan`. The core of this function is the following three steps:

```cpp
uint32 dwToken = CParserBin::NextToken(buf, com, bKey);
dwToken = Parser.NewUserToken(dwToken, com, false);
pBin->m_Tokens.push_back(dwToken);
```
The `NextToken` function parses the tokens that exist in the g_KeyTokens table. The `NewUserToken` function inserts the unknown token into the token table. The value of these unknown tokens is generated by CRC.
Following are the tokens used in the vulkan ray tracing shader:
<p align="center">
    <img src="/resource/vkraytracing/image/tokens_added.png" width="75%" height="75%">
</p>

`RaytracingAccelerationStructure`: used for acceleration structure.
`[shader("raygeneration")]`: used for shader type determination.

Tokens related to shader techniques: Shader technique is unique to Cry Engine. It is similar to the PSO, which indicates the shaders and the render state (depth state, blend state, etc).
```cpp
technique XXX
{
  pass XXX
  {
    VertexShader = XX_VS();
    PixelShader = XX_PS();
  }
}
```

Cry Engine only supports one shader entry for each shader type for each pass. For example, a pass may have two entries (camera close hit and shader close hit shader) for close hit shader type. We need to support multi-entry shader type.

```cpp
technique RayTracingTest
{
  pass p0
  {
    RayGenShaders = {RayGenMain()};
    HitGroupShaders = {ClostHitMain()};
    MissShaders = {MissMain()};
  }
}
```

## Parse

Cry Engine parses the dummy shader first and skips all tokens except those related to the "technique". During this process, Cry Engine enumerates all the shaders in the public techniques.Following are the tokens used in the dummy shader parser:
```cpp
FX_TOKEN(RaytracingAccelerationStructure)
FX_TOKEN(shader)
FX_TOKEN(raygeneration)
FX_TOKEN(closesthit)
FX_TOKEN(miss)
FX_TOKEN(RayGenShaders)
FX_TOKEN(HitGroupShaders)
FX_TOKEN(MissShaders)
FX_TOKEN(vk)
FX_TOKEN(binding)
```
Parse's core function is `ParseObject`, which consists mainly of two steps. The first step is `GetNextToken`, where the next token is obtained. In addition, the more important part of Parse is to parse code fragments, such as preprocessed fragments or function code fragments. It also stores the function name, which is used to find the shader name later when parsing tech. In after parsing the code fragment, can be in ParseObject parse all code snippets of all statements, the Token is divided into the Name/Assign/Value/Data/Annotations type, These sub-SParserFrame in turn form a large ParserFrame based on, for example, semicolons, meaning that a code fragment contains multiple parse fragments, a `CF` (Code Fragment) may be a function, and a `PF` (Parse Fragment) may be an assignment or declaration statement.

And then re-initialize. The `mfLoadDefaultSystemShaders` function loads shader: `sLoadShader (" RayTracingTestShader "s shRayTracingTest)`; And through the `RT_ParseShaderCShaderManBin::ParseBinFX` parsed Tokens, such as `texture`, `float`, `struct`, can parse and load or create a shader.

When parsing technique, CE does not support ray tracing shader. We added support for ray tracing shader in the `ParseBinFX_Technique_Pass_LoadShaders_RayTracing` function:
```cpp
		TArray<uint32> SHData(0, SHDataBuffer.Size());
		SHData.Copy(SHDataBuffer.GetElements(), SHDataBuffer.size());

		CHWShader* pSH = NULL;
		bool bValidShader = false;

		if (bRes && (!CParserBin::m_bParseFX || !SHData.empty() || szName[0] == '$'))
		{
			char str[1024];
			cry_sprintf(str, "%s@%s", Parser.m_pCurShader->m_NameShader.c_str(), szName);
			pSH = CHWShader::mfForName(str, Parser.m_pCurShader->m_NameFile, Parser.m_pCurShader->m_CRC32, szName, eSHClass, SHData, Parser.m_TokenTable, dwSHType, Parser.m_pCurShader, nGenMask, Parser.m_pCurShader->m_nMaskGenFX);
		}
		if (pSH)
		{
			bValidShader = true;

			if (CParserBin::PlatformSupportsRayTracingShaders() && eSHClass == eHWSC_RayGen)
				pPass->m_RGShaders.push_back(pSH);
			else if (CParserBin::PlatformSupportsRayTracingShaders() && eSHClass == eHWSC_HitGroup)
				pPass->m_HGShaders.push_back(pSH);
			else if (CParserBin::PlatformSupportsRayTracingShaders() && eSHClass == eHWSC_RayMiss)
				pPass->m_RMShaders.push_back(pSH);
			else
				CryLog("Unsupported/unrecognised shader: %s[%d]", pSH->m_Name.c_str(), eSHClass);
		}
```

We added support for shader types with multi shader entry:
```cpp
technique BindlessRayTracingTestTech
{
  pass p0
  {
    // other ray tracing shaders
    MissShaders = {MissMain(),ShadowMiassMain()};
  }
}
```
The shader entries are split by the comma token:
```cpp
if (pTokens[nCur] == eT_comma)
{
	nCur++;
}
```

CE technique supports only one shader, while the ray tracing shader consists of a series of shader tables. We need to add the parse and loading functions of shader tables specifically. We specify that all shaders of the same class in raytracing are placed in an array by {}, and then begin parsing shader binding table by `{` token. At the same time, `SShaderPass` is extended to support multiple stores for each shader
```cpp
struct SShaderPass
{
  CHWShader*  m_VShader;        // Pointer to the vertex shader for the current pass
  //...... Other shader s

  std::vector<CHWShader*> m_RGShaders; // Pointers to ray gen shader
}
```
Stored in an array after the `CHWShader` is parsed and created:
```cpp
pPass->m_RGShaders.push_back(pSH);
```

## Cross Compile
The next step is to create the shader (while creating the PSO), first get the shader from technique, and then check whether the shader is active based on the shader information. If it is not in the active state (CHWShader_D3D::mfActivate), `mfCompileHLSL` is called to trigger shader compilation. For debugging purposes, We've turned off asynchronous shader compiling for CE: `CV_r_shadersasynccompiling`

Ray tracing shader target is different from common shaders:
>These shaders are functions compiled into a library, with target model lib_6_3, and identified by an attribute [shader("shadertype")] on the shader function

Set target to `lib 6_3`:
```cpp
	case eHWSC_RayGen:
	case eHWSC_RayMiss:
	case eHWSC_HitGroup:
		if (CParserBin::PlatformSupportsRayTracingShaders())
		{
			szProfile = "lib_6_3";
		}
		break;
```
What's more, the DXC version of CE is older, and we need to upgrade it to the new version. In addition, you need to upgrade the version of spirv cross, otherwise shader reflection process will crash.

In addition, we need to revamp the reflection part of CE-spirv to support spirv's reflection. Once we have the compiled data we need to create vulkan raytracing shader in `mfUploadHW` function.
```cpp
/*
BwAAAGxpYl82XzMACgAAAFJheUdlbk1haW4AAAEDHA0AAEBAHA0AAEBBHA0AAMBAAAAAAAAAAAA=
*/

struct FBasicRayData
{
  float3 Origin;
  uint Mask;
  float3 Direction;
  float TFar;
};
struct FDefaultPayload
{
  float HitT;
  uint PrimitiveIndex;
  uint InstanceIndex;
  float2 Barycentrics;
  uint InstanceID;
};
RaytracingAccelerationStructure TLAS:register(t0);
StructuredBuffer<FBasicRayData>Rays:register(t1);
RWStructuredBuffer<uint>OcclusionOutput:register(u0);
[shader("raygeneration")]void RayGenMain()
{
  const uint RayIndex=DispatchRaysIndex().x;
  FBasicRayData InputRay=Rays[RayIndex];
  RayDesc Ray;
  Ray.Origin=InputRay.Origin;
  Ray.Direction=InputRay.Direction;
  Ray.TMin=0.0f;
  Ray.TMax=InputRay.TFar;
  uint RayFlags=0|RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH|RAY_FLAG_FORCE_OPAQUE|RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
  const uint InstanceInclusionMask=0x01;
  FDefaultPayload Payload=(FDefaultPayload)0;
  TraceRay(TLAS,RayFlags,InstanceInclusionMask,0,2,0,Ray,Payload);
  OcclusionOutput[RayIndex]=(Payload.HitT>0)?~0:0;
}
```
The above is the input of the hlsl shader after parsing.  After compiling by DXC, we get the SPV result. We went through the spirv-cross to convert it to GLSL, and found that a conversion error occurred. After mapping, register t0 and u0 in HLSL have the same binding point:

```cpp
layout(set = 0, binding = 0, std430) buffer type_RWStructuredBuffer_uint
{
    uint _m0[];
} OcclusionOutput;

layout(set = 0, binding = 0) uniform accelerationStructureEXT TLAS;
```
Here’s why the problem occurs:
<p align="center">
    <img src="/resource/vkraytracing/image/confict.png" width="85%" height="85%">
</p>

We solved the problem with [<u>**[this blog]**</u>](https://developer.nvidia.com/blog/bringing-hlsl-ray-tracing-to-vulkan/):
```cpp
[[vk::binding(0,0)]] RaytracingAccelerationStructure topLevelAS;
```
It also needs  to tweak the parse section of the code to skip `[[vk::xxx]]`
```cpp
if (nToken == eT_br_sq_1)
{
	if (m_CurFrame.m_nCurToken + 1 <= m_CurFrame.m_nLastToken)
	{
		if (pTokens[m_CurFrame.m_nCurToken + 1] == eT_br_sq_1)
		{
			int32 nLast1 = m_CurFrame.m_nCurToken + 1;
			while (nLast1 <= m_CurFrame.m_nLastToken && pTokens[nLast1] != eT_br_sq_2)
			{
				nLast1++;
			}
			nLast1++;
			if (nLast1 <= m_CurFrame.m_nLastToken && pTokens[nLast1] == eT_br_sq_2)
			{
				SCodeFragment Fr;
				Fr.m_eType = eFT_StorageClass;
				Fr.m_nFirstToken = m_CurFrame.m_nCurToken;
				m_CurFrame.m_nCurToken = nLast1 + 1;
				while (pTokens[m_CurFrame.m_nCurToken] != eT_semicolumn)
				{
					if (m_CurFrame.m_nCurToken + 1 == nTokensSize)
						break;
					m_CurFrame.m_nCurToken++;
				}
				Fr.m_nLastToken = m_CurFrame.m_nCurToken++;
				Fr.m_dwName = pTokens[Fr.m_nLastToken - 1];
				m_CodeFragments.push_back(Fr);
				continue;
			}
		}
	}
}
```
# Ray Tracing Pipeline
Cry Engine has three render passes: `CComputeRenderPass`, `CPrimitiveRenderPass` and `CSceneRenderPass`. We added a fourth render pass: 'CrayTracingRenderPass'.

## Compile RenderPass
When the PSO or other resources are dirty, Cry Engine rebuilds the resource in the `Compile` function. We create the PSO and shader binding table in the `Compile` function.

## Resource Set

In the cry engine, the slot information is specified in the resource set step. For example, `SetTexture` with slot 2 corresponds to `t2` in HLSL.

```cpp
inline void CRayTracingRenderPass::SetBuffer(uint32 slot, CGpuBuffer* pBuffer)
{
	m_resourceDesc.SetBuffer(slot, pBuffer, EDefaultResourceViews::Default, EShaderStage_RayTracing);
}
```
`SResourceBindPoint` stores slot index, slot type and shader stages. These data are stored in `uint8` format and packed into a `uint32` variable thar will be used in state cache. `SResourceBinding` stores the resource and its view.
```cpp
inline CDeviceResourceSetDesc::EDirtyFlags CDeviceResourceSetDesc::SetBuffer(int shaderSlot, CGpuBuffer* pBuffer, ResourceViewHandle hView, ::EShaderStage shaderStages)
{
	SResourceBinding resource(pBuffer, hView);
	SResourceBindPoint bindPoint(resource, shaderSlot, shaderStages);

	return UpdateResource<SResourceBinding::EResourceType::Buffer>(bindPoint, resource);
}
```
When the resource layout is dirty, the layout needs to be rebuilt.

```cpp
if (dirtyMask & (eDirty_Technique | eDirty_ResourceLayout))
{
	int bindSlot = 0;
	SDeviceResourceLayoutDesc resourceLayoutDesc;
	resourceLayoutDesc.m_needBindlessLayout = m_needBindless;
	resourceLayoutDesc.SetResourceSet(bindSlot++, m_resourceDesc);
	m_pResourceLayout = GetDeviceObjectFactory().CreateResourceLayout(resourceLayoutDesc);

	if (!m_pResourceLayout)
		return (EDirtyFlags)(m_dirtyMask |= revertMask);
}
```

Cry engine stores layout information into the cache through `mfInsertNewCombination` and decodes it during engine initialization. We extended 'mfInitShadersCache' and `GetEncodedResourceLayoutSize` in order to support ray tracing layout encoding.

During the engine initializes the layout, the cry engine decodes the binding information of the descriptor set through `EncodeDescriptorSet`. It includes the binding type and binding stage. This is used to map the DX binding to the Vulkan binding in the latter shader compiling process. We extended `GetShaderStageFlags` to support ray tracing. Cry engine uses the 6 bits of the uint8 to store the shader stage, which is not enough for ray tracing shader. We extended it to `uint32` format to support RayGen/Miss/Hit shader stages.
```cpp
uint8* stagesByte = (uint8*)(&stages);
for (uint32 index = 0; index < 4; index++)
{
	result.push_back(stagesByte[index]);
}
```
We need to pass in this hash in the shader compiling function `GetRayTracingShaderInstanceInfo` so that we can look up the layout later.

Add acceleration descriptor and descriptor pool support:
```cpp
poolSizes[8].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
poolSizes[8].descriptorCount = accelerationStructureCount;
```
```cpp
else if (uint8(bindPoint.flags & SResourceBindPoint::EFlags::IsAccelerationStructured))
{
	return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
}
```
## PSO Creation
New structures related to PSO creation:
1.`CDeviceRayTracingPSODesc`: Contains construction information.
2.`CDeviceRayTracingPSO`: Stores ray tracing PSO.
3.`m_RayTracingPsoCache`: Pipeline state cache.

Obtain and process the shader from the shader cache during PSO creation. It contains the following steps:
1.Get entry function name
```cpp
entryPointName.push_back(rayGenInfo.pHwShader->m_EntryFunc);
```
2.Get the Vulkan shader module and create a shader stage. All stages are stored in a std::vector of VkPipelineShaderStageCreateInfo objects. At this step, indices within this vector will be used as unique identifiers for the shaders.
```cpp
shaderStage.module = reinterpret_cast<NCryVulkan::CShader*>(rayGenInfo.pDeviceShader)->GetVulkanShader();
shaderStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
shaderStage.pName = entryPointName.back().data();
shaderStages.push_back(shaderStage);
```
3.Create the shader group. Shader groups specify the shader stage index. Ray generation and miss shaders are called `general` shaders. In this case the type is `VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR`, and only the generalShader member of the structure is filled
```cpp
VkRayTracingShaderGroupCreateInfoKHR shaderGroup = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
shaderGroup.generalShader = shaderStages.size() - 1;//store general shader index
shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
shaderGroups.push_back(shaderGroup);
```
## Shader Binding Table

A shader binding table is a resource which establishes the relationship between the ray tracing pipeline and the acceleration structures that were built for the ray tracing pipeline. It indicates the shaders that operate on each geometry in an acceleration structure. In addition, it contains the resources accessed by each shader, including indices of textures, buffer device addresses, and constants. The application allocates and manages shader binding tables as VkBuffer objects

Each entry in the shader binding table consists of shaderGroupHandleSize bytes of data, either as queried by vkGetRayTracingShaderGroupHandlesKHR to refer to those specified shaders, or all zeros to refer to a zero shader group.
```cpp
uint32 nHandleDataSize = nHandleCount * nHandleSize;
std::vector<uint8> handleData;
handleData.resize(nHandleDataSize);

CRY_VERIFY(Extensions::KHR_ray_tracing_pipeline::vkGetRayTracingShaderGroupHandlesKHR(GetDevice()->GetVkDevice(), m_pipeline,0, nHandleCount, nHandleDataSize, handleData.data()) == VK_SUCCESS);
```
The SBT is a collection of up to four arrays containing the handles of the shader groups used in the ray tracing pipeline, one array for each of the ray generation, miss, hit and callable (not used here) shader groups

<p align="center">
    <img src="/resource/vkraytracing/image/sbt_0.png" width="60%" height="60%">
</p>

We will ensure that all starting groups start with an address aligned to shaderGroupBaseAlignment and that each entry in the group is aligned to shaderGroupHandleAlignment bytes. All group entries are aligned with shaderGroupHandleAlignment.
```cpp
m_sRayTracingSBT.m_rayGenRegion.stride = alignedValue(nHandleAlign, nBaseAlign);
m_sRayTracingSBT.m_rayGenRegion.size = m_sRayTracingSBT.m_rayGenRegion.stride;// The size member of pRayGenShaderBindingTable must be equal to its stride member

m_sRayTracingSBT.m_rayMissRegion.stride = nHandleAlign;
m_sRayTracingSBT.m_rayMissRegion.size = alignedValue(nRayMissCount * nHandleAlign, nBaseAlign);

m_sRayTracingSBT.m_hitGroupRegion.stride = nHandleAlign;
m_sRayTracingSBT.m_hitGroupRegion.size = alignedValue(nHitGroupCount * nHandleAlign, nBaseAlign);

m_sRayTracingSBT.m_callAbleRegion.stride = nHandleAlign;
m_sRayTracingSBT.m_callAbleRegion.size = 0;
```
In the next section, we store the device address of each shader group.
```cpp
m_sRayTracingSBT.m_rayGenRegion.deviceAddress = sbtAddress;
m_sRayTracingSBT.m_rayMissRegion.deviceAddress = sbtAddress + m_sRayTracingSBT.m_rayGenRegion.size;
m_sRayTracingSBT.m_hitGroupRegion.deviceAddress = sbtAddress + m_sRayTracingSBT.m_rayGenRegion.size + m_sRayTracingSBT.m_rayMissRegion.size;
m_sRayTracingSBT.m_callAbleRegion.deviceAddress = sbtAddress + m_sRayTracingSBT.m_rayGenRegion.size + m_sRayTracingSBT.m_rayMissRegion.size + m_sRayTracingSBT.m_hitGroupRegion.size;
```

The shader binding tables to use in a ray tracing pipeline are passed to the vkCmdTraceRaysNV, vkCmdTraceRaysKHR, or vkCmdTraceRaysIndirectKHR commands. Shader binding tables are read-only in shaders that are executing on the ray tracing pipeline.
```cpp
Extensions::KHR_ray_tracing_pipeline::vkCmdTraceRaysKHR(GetVKCommandList()->GetVkCommandList(), 
	&pVkPipeline->m_sRayTracingSBT.m_rayGenRegion, 
	&pVkPipeline->m_sRayTracingSBT.m_rayMissRegion, 
	&pVkPipeline->m_sRayTracingSBT.m_hitGroupRegion, 
	&pVkPipeline->m_sRayTracingSBT.m_callAbleRegion,
	width, height, 1);
```
# Bindless Ray Tracing

>The limited nature of the binding slots meant that programs could typically only bind the exact set of resources that would be accessed by a particular shader program, which would often have to be done before every draw or dispatch. The CPU-driven nature of binding demanded that a shader's required resources had to be statically known after compilation, which naturally led to inherent restrictions on the complexity of a shader program.

>As ray tracing on the GPU started to gain traction, the classic binding model reached its breaking point. Ray tracing tends to be an inherently global process: one shader program might launch rays that could potentially interact with every material in the scene. This is largely incompatible with the notion of having the CPU bind a fxed set of resources prior to dispatch.

>Fortunately, newer GPUs and APIs no longer suffer from the same limitations. Bindless techniques effectively provide shader programs with full global access to the full set of textures and buffers that are present on the GPU. Instead of requiring the CPU to bind a view for each individual resource, shaders can instead access an individual resource using a simple 32-bit index that can be freely embedded in user-defned data structures.
>------ From Ray Tracing Gems:USING BINDLESS RESOURCES WITH DIRECTX RAYTRACING

Create a descriptor pool to store the descriptors. The size of the bindless descriptor pool is as large as possible to ensure that we can store all the descriptors used in ray training. When creating the Descriptor Pool, we need to add the flag `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT` to ensure descriptor sets allocated from this pool can include bindings with the `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT` bit set
```cpp
void CDeviceBindlessDescriptorManager_Vulkan::CreateBindlessDescriptorPool()
{
	// allocate descriptor pool
	const uint32 setCount = 65535;
	const uint32 sampledImageCount = 32 * 65536;
	const uint32 storageImageCount = 1 * 65536;
	const uint32 uniformBufferCount = 1 * 65536;
	const uint32 uniformBufferDynamicCount = 4 * 65536;
	const uint32 storageBufferCount = 1 * 65536;
	const uint32 uniformTexelBufferCount = 8192;
	const uint32 storageTexelBufferCount = 8192;
	const uint32 samplerCount = 2 * 65536;
	const uint32 accelerationStructureCount = 32 * 1024;

	VkDescriptorPoolSize poolSizes[9];

	poolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	poolSizes[0].descriptorCount = sampledImageCount;

	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[1].descriptorCount = storageImageCount;

	poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[2].descriptorCount = uniformBufferCount;

	poolSizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	poolSizes[3].descriptorCount = uniformBufferDynamicCount;

	poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[4].descriptorCount = storageBufferCount;

	poolSizes[5].type = VK_DESCRIPTOR_TYPE_SAMPLER;
	poolSizes[5].descriptorCount = samplerCount;

	poolSizes[6].type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
	poolSizes[6].descriptorCount = uniformTexelBufferCount;

	poolSizes[7].type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
	poolSizes[7].descriptorCount = storageTexelBufferCount;

	poolSizes[8].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	poolSizes[8].descriptorCount = accelerationStructureCount;

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
	descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCreateInfo.pNext = nullptr;

	descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;//TanGram:BINDLESS

	descriptorPoolCreateInfo.maxSets = setCount;
	descriptorPoolCreateInfo.poolSizeCount = CRY_ARRAY_COUNT(poolSizes);
	descriptorPoolCreateInfo.pPoolSizes = poolSizes;

	CRY_ASSERT(vkCreateDescriptorPool(GetDevice()->GetVkDevice(), &descriptorPoolCreateInfo, nullptr, &m_bindlessDescriptorPool) == VK_SUCCESS);
}
```
Create the Descriptor Set Layout with at least the flags `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT` and `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT`. 

Descriptor binding flag `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT`: This flag indicates that if we update descriptor after it is bound (i.e using `vkBindDescriptorSets`), the command submission will use the most recently updated version of the descriptor set and most importantly, the update will NOT invalidate the command buffer. 

Descriptor binding flag `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT`: This flag indicates that descriptor set does not need to have valid descriptors in them as long as the invalid descriptors are not accessed during shader execution.

```cpp
	VkDescriptorSetLayoutBindingFlagsCreateInfo descriptorSetLayoutBindingFlagsCreateInfo;
	descriptorSetLayoutBindingFlagsCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
	descriptorSetLayoutBindingFlagsCreateInfo.bindingCount = EBindlessDescriptorBindingType::e_bdbtNum;
	descriptorSetLayoutBindingFlagsCreateInfo.pBindingFlags = bindingFlags.data();

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();
	descriptorSetLayoutCreateInfo.bindingCount = EBindlessDescriptorBindingType::e_bdbtNum;
	descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
	//descriptorSetLayoutCreateInfo.flags = 0;
	descriptorSetLayoutCreateInfo.pNext = &descriptorSetLayoutBindingFlagsCreateInfo;

	CRY_ASSERT(vkCreateDescriptorSetLayout(GetDevice()->GetVkDevice(), &descriptorSetLayoutCreateInfo, nullptr, &m_bindlessDescriptorSetLayout) == VK_SUCCESS);
```

Then, create the actual descriptor set from the bindless pool:
```cpp
	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
	descriptorSetAllocateInfo.descriptorPool = m_bindlessDescriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &m_bindlessDescriptorSetLayout;

	VkDescriptorSetVariableDescriptorCountAllocateInfoEXT descriptorSetVariableDescriptorCountAllocateInfoEXT;
	descriptorSetVariableDescriptorCountAllocateInfoEXT = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT };
	uint32 maxAllocatableCount = bindlessDescriptorCounrPerType - 1;
	descriptorSetVariableDescriptorCountAllocateInfoEXT.descriptorSetCount = 1;
	// This number is the max allocatable count
	descriptorSetVariableDescriptorCountAllocateInfoEXT.pDescriptorCounts = &maxAllocatableCount;
	descriptorSetAllocateInfo.pNext = &descriptorSetVariableDescriptorCountAllocateInfoEXT;

	CRY_ASSERT(vkAllocateDescriptorSets(GetDevice()->GetVkDevice(), &descriptorSetAllocateInfo, &m_bindlessDescriptorSet) == VK_SUCCESS);
```
If we want to use a buffer as a bindless resource, we need to obtain the bindless index from the free list array managed by ourselves. We also need to update the descriptors in the global descriptor pool.

```cpp
uint32 CDeviceObjectFactory::SetBindlessStorageBufferImpl(const CDeviceInputStream* DeviceStreaming, uint32 bindingIndex)
{
	CRY_ASSERT(bindingIndex >= 2 && bindingIndex <= 5);

	CDeviceBindlessDescriptorManager_Vulkan* pDeviceBindlessDescriptorManager = static_cast<CDeviceBindlessDescriptorManager_Vulkan*>(m_pDeviceBindlessDescriptorManager);
	
	buffer_size_t offset;
	CBufferResource* const pActualBuffer = gcpRendD3D.m_DevBufMan.GetD3D(((SStreamInfo*)DeviceStreaming)->hStream, &offset);
	VkBuffer buffer = pActualBuffer->GetHandle();

	SBufferBindingState& storageufferBindingState = pDeviceBindlessDescriptorManager->m_BufferBindingState[bindingIndex];

	uint32 currentFreeIndex = storageufferBindingState.m_currentFreeIndex;
	storageufferBindingState.m_currentFreeIndex = storageufferBindingState.m_freeIndexArray[storageufferBindingState.m_currentFreeIndex];

	uint32 bufferSize = storageufferBindingState.m_buffers.size();
	if (currentFreeIndex >= bufferSize)
	{
		storageufferBindingState.m_buffers.resize(alignedValue(currentFreeIndex + 1, 1024));
		storageufferBindingState.m_bufferInfos.resize(alignedValue(currentFreeIndex + 1, 1024));
	}
	storageufferBindingState.m_buffers[currentFreeIndex] = buffer;
	
	
	VkDescriptorBufferInfo descriptorBufferInfo;
	//range is the size in bytes that is used for this descriptor update, or VK_WHOLE_SIZE to use the range from offset to the end of the buffer.
	descriptorBufferInfo.range = VK_WHOLE_SIZE;
	descriptorBufferInfo.buffer = storageufferBindingState.m_buffers[currentFreeIndex];
	descriptorBufferInfo.offset = offset;
	storageufferBindingState.m_bufferInfos[currentFreeIndex] = descriptorBufferInfo;

	VkWriteDescriptorSet writeDescriptorSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
	writeDescriptorSet.dstSet = pDeviceBindlessDescriptorManager->m_bindlessDescriptorSet;
	writeDescriptorSet.dstBinding = bindingIndex;
	writeDescriptorSet.dstArrayElement = currentFreeIndex;
	writeDescriptorSet.descriptorType = ConvertToDescriptorType(bindingIndex);
	writeDescriptorSet.pBufferInfo = &storageufferBindingState.m_bufferInfos[currentFreeIndex];
	writeDescriptorSet.descriptorCount = 1;

	pDeviceBindlessDescriptorManager->m_descriptorSetWrites.emplace_back(writeDescriptorSet);
	pDeviceBindlessDescriptorManager->m_isUpdateDescriptor = false;

	return currentFreeIndex;
}
```
Bind the bindless descriptor set during ray tracing dispatch if we use bindless descriptor set.

```cpp
	if (pVkLayout->m_needBindless)
	{
		vkCmdBindDescriptorSets(
			GetVKCommandList()->GetVkCommandList(),
			VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
			pVkLayout->GetVkPipelineLayout(),
			1, /*TODO:FixMe*/
			1,
			&pDeviceBindlessDescriptorManager->m_bindlessDescriptorSet, 0, nullptr);
	}
```
result in cry engine:
<p align="center">
    <img src="/resource/vkraytracing/image/result.png" width="75%" height="75%">
</p>

[<u>**Bindless Vulkan RayTracing In Cry Engine Source Code**</u>](https://github.com/ShawnTSH1229/VkRtInCryEngine)


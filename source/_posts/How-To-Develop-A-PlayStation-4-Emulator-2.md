---
title: How To Develop A PlayStation 4 Emulator(2)
date: 2024-07-07 21:54:02
tags:
index_img: /resource/pset/image/PS4Emulator.png
---

# Overview

AMD Graphics driver has four levels. The first level is GNM application level. This level is implemented by the game engine or the game developer. After that, the user mode driver translates the GNM API into PM4 packets. PM4 packets represent API commands in a way that GPUs can execute. The user mode driver library is PS4's built-in library. We have implemented a custom GNM driver to replace the built-in library. Then, Operating System gets the prepared command buffer from the user mode driver. Hands it over to the kernel mode driver. Finally, the kernel mode driver uses a ring buffer to communicate with the GPU. We implemented a command processor to process PM4 packets in order to simulate the kernel mode driver.

Our command processor is designed to simulate setting GPU registers. These GPU registers include user data registers, blend state registers, depth state registers etc.  And we have implemented an AMD ISA converter to parse AMD ISA and translate it into Spirv. Finally, with these GPU registers and spirv have been prepared, we translate them into Vulkan API and rendering the game.

# Graphics Libraries

## Graphics Driver Overview

AMD Graphics driver has four levels. The first level is GNM application level. This level is implemented by the game engine or the game developer. Game developers call these GNM commands to prepare GPU resources, such as buffer, PSO and draw commands.

<p align="center">
    <img src="/resource/pset/image/psdriver_gnm.png" width="75%" height="75%">
</p>

After that, the user mode driver translates the GNM API into PM4 packets. PM4 packets represent API commands in a way that GPUs can execute. The translation library is called the Platform Abstraction Library (Linux driver is open source on Github). It translates the Graphics APIs (Vulkan/GNM) into PM4 packets. PAL is a shared component that is designed to encapsulate certain hardware and OS-specific programming details for many of AMD's 3D and compute drivers.

<p align="center">
    <img src="/resource/pset/image/psdriver_user_mode_driver.png" width="75%" height="75%">
</p>

Here is a PM4 example. The `header` member is the opcode of `draw index 2`. The `indexBaseLo` and `indexBaseHi` are the lower and higher part of the GPU address. As we mentioned above, the GPU memory and the CPU memory are viewed as the same in our emulator, which means that we can get the resource from the allocated memory. The user mode driver will generate the PM4 packets and write them to the command buffer.

<p align="center">
    <img src="/resource/pset/image/psdriver_pm4.png" width="65%" height="65%">
</p>

Then, Operating System gets the prepared command buffer from the user mode driver. Hands it over to the kernel mode driver. Ensure that the user mode driver cannot hang the whole system.

<p align="center">
    <img src="/resource/pset/image/psdriver_os.png" width="75%" height="75%">
</p>

Finally, the kernel mode driver uses a ring buffer to communicate with the GPU. It contains addresses to command buffers. CPU increments the write pointer. GPU increments the read pointer.

<p align="center">
    <img src="/resource/pset/image/psdriver_kernel_mode.png" width="75%" height="75%">
</p>

There are two steps that we should do. The first step is to implement a custom GNM driver that translates the GNM API into PM4 packets. The second step is to implement a kernel mode driver to simulate register setting. The GNM driver and the kernel mode driver are running on the CPU. We also need to implement a command processor to process PM4 packets and translate them into the Vulkan API. The command processor runs on the GPU, which is different from the driver.

<p align="center">
    <img src="/resource/pset/image/psdriver_what_we_do.png" width="65%" height="65%">
</p>

## GNM Driver

Here is an example of the GNM API implementation. As you can see, it is non-human-readable. You can't understand these magic hexadecimal numbers without AMD's official documentation. 

```pascal
function ps4_sceGnmUpdatePsShader(cmdBuffer:PDWORD;numDwords:DWORD;psRegs:PPsStageRegisters):Integer; SysV_ABI_CDecl;
begin
  cmdBuffer[0]  :=$c0027600;
  cmdBuffer[1]  :=8;
  cmdBuffer[2]  :=psRegs^.m_spiShaderPgmLoPs;
  cmdBuffer[3]  :=psRegs^.m_spiShaderPgmHiPs;
  cmdBuffer[4]  :=$c0027600;
  cmdBuffer[5]  :=10;
  cmdBuffer[6]  :=psRegs^.m_spiShaderPgmRsrc1Ps;
  cmdBuffer[7]  :=psRegs^.m_spiShaderPgmRsrc2Ps;
  cmdBuffer[8]  :=$c0021000;
  cmdBuffer[9]  :=$c01e01c4;
  // ......
end
```
Since the PM4 packets generated from the GNM driver are used in the kernel mode driver, which is also under our control. It means we can add a custom more human-readable format. This format is encoded in the GNM driver and decoded in the kernel mode driver.
We generate the OPcode by the `PM4_HEADER_BUILD` macro. The custom OPcode is marked as `IT_OP_CUS`.
```cpp
param->opcode = PM4_HEADER_BUILD(paramSize, IT_OP_CUS, OP_CUS_DRAW_INDEX);
```
We process the custom OPcode separately in the kernel mode driver for the `IT_OP_CUS` case.

```cpp
void CPtGnmTranslator::ProcessPM4Type3(PM4_PT_TYPE_3_HEADER* pm4Hdr, uint32_t* itBody)
{
	uint32_t opcode = pm4Hdr->opcode;
	switch (opcode)
	{
	case PtGfx::IT_NOP:
	case PtGfx::IT_ACQUIRE_MEM:
	case xxxxxx:
		// ......
		break;
	case IT_OP_CUS:
		ProcessGnmPrivateOp(pm4Hdr, itBody);
		break;
	}
}
```
Here is the implementation of the GNM `DrawIndex` API. We generate the custom PM4 packets and output the result to the command buffer. 
```cpp
int PSET_SYSV_ABI Pset_sceGnmDrawIndex(uint32_t* cmdBuffer, uint32_t numDwords, uint32_t indexCount, const void* indexAddr, uint32_t predAndMod, uint32_t inlineMode)
{
	const uint32_t paramSize = sizeof(GnmCmdDrawIndex) / sizeof(uint32_t);
	GnmCmdDrawIndex* param = (GnmCmdDrawIndex*)cmdBuffer;
	param->opcode = PM4_HEADER_BUILD(paramSize, IT_OP_CUS, OP_CUS_DRAW_INDEX);
	param->indexCount = indexCount;
	param->indexAddr = (uintptr_t)indexAddr;
	param->predAndMod = predAndMod;
	param->inlineMode = inlineMode;
	memset(param->reserved, 0, sizeof(param->reserved));
	return PSET_OK;
}
```
## Kernel Mode Driver

In the kernel mode driver, we decode the PM4 packets from the command buffer and set registers based on the decoded packets.
```cpp
void CPtGnmTranslator::TranslateAndDispatchCmd(const void* commandBuffer, uint32_t commandSize)
{
	//......
	const PM4_HEADER_COMMON* pm4Hdr = reinterpret_cast<const PM4_HEADER_COMMON*>(commandBuffer);
	while (processedCmdBufferLen < commandSize)
	{
		uint32_t pm4Type = pm4Hdr->type;
		//......
		const PM4_HEADER_COMMON* nextPm4Hdr = GetNextPm4(pm4Hdr, processedPm4Count);
		pm4Hdr = nextPm4Hdr;
	}
	//......
}
```

Three types of PM4 command packets are currently defined. They are types 0, 2 and 3. We only parse the PM4 packets with the `PM4_TYPE_3` type.
```cpp
uint32_t pm4Type = pm4Hdr->type;
switch (pm4Type)
{
case PM4_TYPE_0:
case PM4_TYPE_2:
	break;
case PM4_TYPE_3:
	ProcessPM4Type3((PM4_PT_TYPE_3_HEADER*)pm4Hdr, (uint32_t*)(pm4Hdr + 1));
	break;
}
```

A PM4 command packet consists of a packet header, identified by field HEADER, and an information body, identified by IT_BODY, that follows the header. The packet header defines the operations to be carried out by the PM4 micro-engine, and the information body contains the data to be used by the engine in carrying out the operation. Following parts are the packets we will process.

### Constant Engine Packets

`DUMP_CONST_RAM`: Sent only to the Constant Engine to instruct the CE to move data from one or more RAM slot lines to L2 Memory.

`INDIRECT_BUFFER_CONST`: This packet is used for dispatching Constant Indirect Buffers, new in SI. A separate constant buffer allows the CP to process constants ahead of and concurrently with the "draw command buffer"

#### SET_CONTEXT_REG
`SET_CONTEXT_REG`: This packet loads the eight-context-renders tate register data, which is embedded in the packet, into the chip. 

The `REG_OFFSET` field is a DWord-offset from the starting address:
```cpp
struct
{
    unsigned int    regOffset : 16;  ///< offset in DWords from the register base address
    unsigned int    reserved1 : 12;  ///< Program to zero
    unsigned int    index : 4;   ///< Index for UCONFIG/CONTEXT on CI+
};
```
All the render state data in the packet is written to consecutive register addresses beginning at the starting address. The starting address for register data is computed as follows:
```cpp
void CPtGnmTranslator::onSetContextRegs(PM4_PT_TYPE_3_HEADER* pm4Hdr, PtGfx::PM4CMDSETDATA* itBody)
{
	constexpr uint32_t CONTEXT_REG_BASE = 0xA000;
	uint32_t count = pm4Hdr->count;
	for (uint32_t index = 0; index < count; index++)
	{
		uint16_t reg = CONTEXT_REG_BASE + itBody->regOffset + index;
		uint32_t value = *((uint32_t*)(itBody + 1 + index));
		SetContextReg(reg, value);
	}
}
```
Many render states are related to `SET_CONTEXT_REG`. It includes the following states:

render target states:
```cpp
((uint32_t*)GetGpuRegs()->RENDER_TARGET)[reg - PtGfx::mmCB_COLOR0_BASE] = value;
````

blend states:
```cpp
GetGpuRegs()->CB_BLEND_CONTROL[reg - PtGfx::mmCB_BLEND0_CONTROL] = *(PtGfx::CB_BLEND0_CONTROL*)&value;
```

**PA(Primitive Assembler)**, **SC(Scan Converter)** modes control:
```cpp
	case PtGfx::mmPA_SU_SC_MODE_CNTL:
		GetGpuRegs()->SC_MODE_CNTL = *(PtGfx::PA_SU_SC_MODE_CNTL*)&value;
```
Here is the `PA_SU_SC_MODE_CNTL` structure:
```cpp
	union PA_SU_SC_MODE_CNTL {
	struct {
		unsigned int                      CULL_FRONT : 1;
		unsigned int                       CULL_BACK : 1;
		unsigned int                            FACE : 1;
		unsigned int                       POLY_MODE : 2;
		unsigned int            POLYMODE_FRONT_PTYPE : 3;
		unsigned int             POLYMODE_BACK_PTYPE : 3;
		unsigned int        POLY_OFFSET_FRONT_ENABLE : 1;
		unsigned int         POLY_OFFSET_BACK_ENABLE : 1;
		unsigned int         POLY_OFFSET_PARA_ENABLE : 1;
		unsigned int                                 : 2;
		unsigned int        VTX_WINDOW_OFFSET_ENABLE : 1;
		unsigned int                                 : 2;
		unsigned int              PROVOKING_VTX_LAST : 1;
		unsigned int                  PERSP_CORR_DIS : 1;
		unsigned int               MULTI_PRIM_IB_ENA : 1;
		unsigned int                                 : 10;
	} bitfields, bits;
	unsigned int	u32All;
	signed int	i32All;
	float	f32All;
	};
```
##### PA(Primitive Assembler)
Primitive Assembler accumulates vertices that span a triangle and forwards triangles to SC.

<p align="center">
    <img src="/resource/pset/image/PA.png" width="65%" height="65%">
</p>

Primitive Assembly is the stage in the OpenGL rendering pipeline where Primitives are divided into a sequence of individual base primitives. The purpose of the primitive assembly step is to convert a vertex stream into a sequence of base primitives. For example, a primitive which is a line list of 12 vertices needs to generate 11 line base primitives. The order in which primitives are processed is well-defined (in most cases).

##### SC(Scan Converter)
Scan Converter determines pixels covered by each triangle and forwards them to SPI.
<p align="center">
    <img src="/resource/pset/image/SC.png" width="65%" height="65%">
</p>

Scan conversion generates the set of fragments corresponding to each primitive. Each fragment contains window coordinates x, y, and z, a color value, and texture coordinates. The fragment values are generated by interpolating the attributes provided at each vertex in the primitive. Scan conversion is computationally intensive since multiple attribute values must be computed for each fragment. During scan conversion the generated fragments are also tested against the scissor rectangle, and fragments outside the rectangle are discarded

#### SPI(Shader Processor Input)
Shader Processor Input accumulates work items and Sends them in waves to the CU.
```cpp
if (reg >= PtGfx::mmSPI_PS_INPUT_CNTL_0 && reg <= PtGfx::mmSPI_PS_INPUT_CNTL_31)
{
	GetGpuRegs()->SPI.PS.INPUT_CNTL[reg - PtGfx::mmSPI_PS_INPUT_CNTL_0] = *(PtGfx::SPI_PS_INPUT_CNTL_0*)&value;
}
```

<p align="center">
    <img src="/resource/pset/image/SPI.png" width="65%" height="65%">
</p>

##### CB/DB(Color Backend/Depth Backend)
Depth Backend discards occluded fragments based on depth / stencil. Color Backend write colored fragments to render targets.
```cpp
GetGpuRegs()->DEPTH.DEPTH_CONTROL = *(PtGfx::DB_DEPTH_CONTROL*)&value;
GetGpuRegs()->CB_COLOR_CONTROL = *(PtGfx::CB_COLOR_CONTROL*)&value;
```
<p align="center">
    <img src="/resource/pset/image/CB_DB.png" width="65%" height="65%">
</p>

#### SET_SH_REG
This packet updates the shader persistent register state in the SPI, which is embedded in the packet, into the chip. The starting address for register data is computed as follows:
```cpp
void CPtGnmTranslator::OnSetShRegs(PM4_PT_TYPE_3_HEADER* pm4Hdr, PtGfx::PM4CMDSETDATA* itBody)
{
	constexpr uint32_t SH_REG_BASE = 0x2C00;
	uint32_t count = pm4Hdr->count;
	for (uint32_t index = 0; index < count; index++)
	{
		uint16_t reg = SH_REG_BASE + itBody->regOffset + index;
		uint32_t value = *((uint32_t*)(itBody + 1 + index));
		SetShReg(reg, value);
	}
}
```
`SET_SH_REG` is used to set the shader user data. For each shader type, there are 16 user data hardware registers (e.g., SPI_SHADER_USER_DATA_PS_0-15). These 16 32-bit values are the only interface through which the driver can specify SRDs to the shader:

These registers are written with SET_SH_REG PM4 commands in the driver's indirect buffer. 

At shader launch, SPI loads values from these 16 hardware registers into SGPRs 0 - 15 to be read by the shader. The actual number of values loaded by SPI is controlled by the USER_SGPR field in the SPI_SHADER_PGM_RSRC2_xx register.

```cpp
if (reg >= PtGfx::mmSPI_SHADER_USER_DATA_VS_0 && reg <= PtGfx::mmSPI_SHADER_USER_DATA_VS_15)
{
	GetGpuRegs()->SPI.VS.USER_DATA[reg - PtGfx::mmSPI_SHADER_USER_DATA_VS_0] = value;
}
else if(reg >= PtGfx::mmSPI_SHADER_USER_DATA_PS_0 && reg <= PtGfx::mmSPI_SHADER_USER_DATA_PS_15)
{
	GetGpuRegs()->SPI.PS.USER_DATA[reg - PtGfx::mmSPI_SHADER_USER_DATA_PS_0] = value;
}
```

### Draw Related Packets: Graphics Ring

`INDEX_TYPE`: Set the index buffer format.
```cpp
GetGpuRegs()->VGT_DMA.INDEX_TYPE = *(PtGfx::VGT_INDEX_TYPE_MODE*)(itBody + 1);
```
There are three optional index formats: 16 bit, 32 bit and 8 bit index formats.
```cpp
typedef enum VGT_INDEX_TYPE_MODE {
VGT_INDEX_16                             = 0x00000000,
VGT_INDEX_32                             = 0x00000001,
VGT_INDEX_8__VI                          = 0x00000002,
} VGT_INDEX_TYPE_MODE;
```
`NUM_INSTANCES`: NUM_INSTANCES is used to specify the number of instances for the subsequent draw command.
```cpp
GetGpuRegs()->VGT_NUM_INSTANCES = *(PtGfx::VGT_NUM_INSTANCES*)(&value);
```

### Custom PM4 Packets

As we mentioned in the `GNM driver` part, we added a custom PM4 packet format to the GNM API. The opcode of these packets is `IT_OP_CUS`. We process these packets separately.
```cpp
	uint32_t opcode = pm4Hdr->opcode;
	switch (opcode)
	{
	case xxx:
		break;
	case IT_OP_CUS:
		ProcessGnmPrivateOp(pm4Hdr, itBody);
		break;
	}
```
The vertex shader input and pixel shader input are set by custom PM4 packets. We will detail its usage latter.

```cpp
case OP_CUS_SET_VS_SHADER:
{
	GnmCmdSetVSShader* param = (GnmCmdSetVSShader*)pm4Hdr;
	GetGpuRegs()->SPI.VS.LO = param->vsRegs.spiShaderPgmLoVs;
	GetGpuRegs()->SPI.VS.HI = param->vsRegs.spiShaderPgmHiVs;
	GetGpuRegs()->SPI.VS.RSRC1 = *(PtGfx::SPI_SHADER_PGM_RSRC1_VS*)(&param->vsRegs.spiShaderPgmRsrc1Vs);
	GetGpuRegs()->SPI.VS.RSRC2 = *(PtGfx::SPI_SHADER_PGM_RSRC2_VS*)(&param->vsRegs.spiShaderPgmRsrc2Vs);
	GetGpuRegs()->SPI.VS.OUT_CONFIG = *(PtGfx::SPI_VS_OUT_CONFIG*)(&param->vsRegs.spiVsOutConfig);
	GetGpuRegs()->SPI.VS.POS_FORMAT = *(PtGfx::SPI_SHADER_POS_FORMAT*)(&param->vsRegs.spiShaderPosFormat);
	GetGpuRegs()->SPI.VS.OUT_CNTL = *(PtGfx::PA_CL_VS_OUT_CNTL*)(&param->vsRegs.paClVsOutCntl);
	break;
}
case OP_CUS_SET_PS_SHADER:
{
	GnmCmdSetPSShader* param = (GnmCmdSetPSShader*)pm4Hdr;
	GetGpuRegs()->SPI.PS.LO = param->psRegs.spiShaderPgmLoPs;
	GetGpuRegs()->SPI.PS.HI = param->psRegs.spiShaderPgmHiPs;
	GetGpuRegs()->SPI.PS.RSRC1 = *(PtGfx::SPI_SHADER_PGM_RSRC1_PS*)(&param->psRegs.spiShaderPgmRsrc1Ps);
	GetGpuRegs()->SPI.PS.RSRC2 = *(PtGfx::SPI_SHADER_PGM_RSRC2_PS*)(&param->psRegs.spiShaderPgmRsrc2Ps);
	GetGpuRegs()->SPI.PS.Z_FORMAT = *(PtGfx::SPI_SHADER_Z_FORMAT*)(&param->psRegs.spiShaderZFormat);
	GetGpuRegs()->SPI.PS.COL_FORMAT = *(PtGfx::SPI_SHADER_COL_FORMAT*)(&param->psRegs.spiShaderColFormat);
	GetGpuRegs()->SPI.PS.INPUT_ENA = *(PtGfx::SPI_PS_INPUT_ENA*)(&param->psRegs.spiPsInputEna);
	GetGpuRegs()->SPI.PS.INPUT_ADDR = *(PtGfx::SPI_PS_INPUT_ADDR*)(&param->psRegs.spiPsInputAddr);
	GetGpuRegs()->SPI.PS.IN_CONTROL = *(PtGfx::SPI_PS_IN_CONTROL*)(&param->psRegs.spiPsInControl);
	GetGpuRegs()->SPI.PS.BARYC_CNTL = *(PtGfx::SPI_BARYC_CNTL*)(&param->psRegs.spiBarycCntl);
	GetGpuRegs()->SPI.PS.SHADER_CONTROL = *(PtGfx::DB_SHADER_CONTROL*)(&param->psRegs.dbShaderControl);
	GetGpuRegs()->SPI.PS.SHADER_MASK = *(PtGfx::CB_SHADER_MASK*)(&param->psRegs.cbShaderMask);
	break;
}
```
## GPU Translator

After all registers have been set, we can implement a GPU translator that translates GPU registers into Vulkan API.

### Shader Binary Code Layout

The shader binary code we obtained from the game project is represented in AMD ISA format. It is compiled from PSSL (Play Station Shading Language). Here is the shader binary code layout for PS4.

<p align="center">
    <img src="/resource/pset/image/shader_bin_code_layout.png" width="50%" height="50%">
</p>

The shader binary code is starts with the AMD ISA compiled from the PSSL. Following the ISA are the shader usage slots that point to the user data register. We will detail it later. The shader binary information is after shader usage slots. It is an exclusive part for PS4. It contains information about the shader binary code, such as the ISA length, the shader usage slots offset relative to the shader binary information address.

### AMD ISA Parser and Translator

Since the game is rendered using Vulkan, we must translate the AMD ISA into Spirv. Fortunatly, there are many open source tools that can disassemble the AMD ISA, such as the Radeon GPU Analyzer. In PSET4, we rely on the ISA translator developed by FPCS4. The reason we use this translator is that it can converts the ISA into Spirv directly.

Due to the fact that the FPCS4's ISA translator requires some additional information contained within the ISA, we need to implement a simplified ISA parser in order to obtain these details. What's more, the parsed information is also used in the resource binding.

AMD ISA has six types of operation: scalar ALU operations, vector ALU operations, scalar memory operations, vector memory operations, flat memory instructions and data share operations. We take the scalar ALU operation as an example to introduce the instruction format.

Scalar ALU (SALU) instructions operate on a single value per wavefront. These operations consist of 32-bit integer arithmetic and 32- or 64-bit bit-wise operations. The SALU also can perform operations directly on the Program Counter, allowing the program to create a call stack in SGPRs. Many operations also set the Scalar Condition Code bit (SCC) to indicate the result of a comparison, a carry-out, or whether the instruction result was zero.

SALU instructions are encoded in one of five microcode formats, shown below:

<p align="center">
    <img src="/resource/pset/image/salu_format.png" width="70%" height="70%">
</p>

Each of these instruction formats uses some of these fields:
|Field| Description|
|-------|-------|
|OP |Opcode: instruction to be executed.|
|SDST| Destination SGPR.|
|SSRC0| First source operand.|
|SSRC1 |Second source operand.|
|SIMM16 |Signed immediate 16-bit integer constant.|

The instructions begin with two or more bits that are used to determine which instruction the uint32 value belongs to. For example, the SOP1 instruction begins with 1011 1110 1, which is 0xbe800000 in hexadecimal format.
```cpp
#define SQ_ENC_SOP1_BITS                                0xbe800000
#define SQ_ENC_SOP1_FIELD                               0x0000017d
#define SQ_ENC_SOP1_MASK                                0xff800000
```
We get the value of the first 9 bits by `SQ_ENC_SOP1_MASK` and compare it with `SQ_ENC_SOP1_BITS`. If the compare result is true, we can process this instruction as the `SOP1` instruction.

```cpp
if ((token & SSQ_ENC_SOP1_MASK) == SQ_ENC_SOP1_BITS)
{
	const uint32_t instruction = codeReader.ReadU32size();
	return instruction;
}
```
Here is an example:
```cpp
if ((ins & SQ_ENC_SOP1_MASK) == SQ_ENC_SOP1_BITS)
{
	PtGfx::SQ_SOP1 SOP1 = *(PtGfx::SQ_SOP1*)(&ins);
	if (uint32_t(SOP1.bitfields.OP) == SQ_S_SETPC_B64)
	{
		// do some thing
	}
}
```
### Vertex Shader State

The shader binary code contains shader usage slots. It stores the index that point to the user data reigisters. Each shader type has 16 user data registers. The content of the user data registers depends on the shader usage slot type. Generally, it stores the resource descriptor address of the shader input resources. The resource descriptor contains resource information such as the resource GPU address, resource format and stride.

<p align="center">
    <img src="/resource/pset/image/resource_desc.png" width="50%" height="50%">
</p>

The first step is to get the shader register index from the shader binary code for the resources used by the vertex shader. The base address of the shader usage slot is calculated based on shader binary information.

```cpp
int32_t CGsISAProcessor::GetUsageRegisterIndex(EShaderInputUsage usgae)
{
	SInputUsageSlot* usageSlot = GetShaderSlot();
	if (usageSlot != nullptr)
	{
		int32_t numUsageSlot = GetShaderInfo(m_base)->m_numInputUsageSlots;
		for (uint32_t index = 0; index < numUsageSlot; index++)
		{
			if (usgae == usageSlot[index].m_usageType) { return usageSlot[index].m_startRegister; };
		}
	}
	return -1;
}
```

#### Set Vertex Shader Constant Buffer

The constant buffer usage type is `kShaderInputUsageImmConstBuffer`. We can get the user data register index by the `GetUsageRegisterIndex` function.  
```cpp
pVertexShader->m_cbvStartRegIdnex = isaProcessor.GetUsageRegisterIndex(kShaderInputUsageImmConstBuffer);
```
Then, retrieve the constant buffer resource descriptor's GPU address from the user data register.
```cpp
const uint32_t* cbvDesc = reinterpret_cast<uint32_t*>(&GetGpuRegs()->SPI.VS.USER_DATA[pVertexShader->m_cbvStartRegIdnex + index * ShaderConstantDwordSize::kDwordSizeConstantBuffer]);
const CBufferResourceDesc* cbDesc = reinterpret_cast<const CBufferResourceDesc*>(cbvDesc);
```
The shader buffer resource descriptor has four words. The first word stores the lower part of the buffer's byte base address.
```cpp
union SQ_BUF_RSRC_WORD0 
{
	unsigned int BASE_ADDRESS : 32;
};
```
The second word has four members:
`BASE_ADDRESS_HI`: The higher part of the buffer byte base address.
`STRIDE`: Stride, in bytes. [0..2048]
`CACHE_SWIZZLE `: buffer access. optionally swizzle TC L1 cache banks.
`SWIZZLE_ENABLE`: Cache Swizzle Array-Of-Structures according to stride, index_stride and element_size; else linear.

```cpp
union SQ_BUF_RSRC_WORD1 
{
		unsigned int                 BASE_ADDRESS_HI : 16;
		unsigned int                          STRIDE : 14;
		unsigned int                   CACHE_SWIZZLE : 1;
		unsigned int                  SWIZZLE_ENABLE : 1;
};
```
The third word stores the number of records in a buffer. Each record is `STRIDE` bytes. We will detail the last word later.
```cpp
union SQ_BUF_RSRC_WORD2 
{
	unsigned int                     NUM_RECORDS : 32;
};
```
The buffer's GPU address is calculated from word0 and word1.

```cpp
void* CBufferResourceDesc::GetBaseAddress()const
{
    uintptr_t baseAddr = m_bufferSrd.word1.bitfields.BASE_ADDRESS_HI;
    baseAddr <<= 32;
    baseAddr |= m_bufferSrd.word0.bitfields.BASE_ADDRESS;
    return (void*)baseAddr;
}
```
We calculate the buffer size by multiplying the element stride and the number of the elements.

```cpp
uint32_t CBufferResourceDesc::GetSize() const
{
    uint32_t stride = m_bufferSrd.word1.bitfields.STRIDE;
    uint32_t numElements = m_bufferSrd.word2.bitfields.NUM_RECORDS;
    return stride ? numElements * stride : numElements;
}
```
The constant buffer can now be created and set now that all the resources needed have been prepared.
```cpp
pVertexShader->m_pConstantBuffers[index] = RHICreateBuffer(cbDesc->GetBaseAddress(), cbDesc->GetSize(), 1, RHIBU_CB);
gRHICommandList.RHISetConstantBuffer(pVertexShader->m_pConstantBuffers[index].get(), index);
```

### Vertex Attributes And AMD GCN Assembly

We should get the vertex attributes information before setting the vertex buffer. The vertex attributes are obtained from the fetch shader. Here is an example of the fetch shader ISA. The fetch shader is used to load the vertex element from the vertex buffer, which means that we can obtain the vertex attributes by parsing shader ISA.
```cpp
s_load_dwordx4 s[8:11], s[2:3], 0x00                   
s_load_dwordx4 s[12:15], s[2:3], 0x04                  
s_load_dwordx4 s[16:19], s[2:3], 0x08                  
s_waitcnt     lgkmcnt(0)                               
buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen   
buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen  
buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen  
s_waitcnt     0                                        
s_setpc_b64   s[0:1]    
```

In this example, the hardware loads the buffer address into scalar registers 8 - 19 from scalar registers 2 - 3 by the 's_load_dwordx4' instruction. The registers 2 - 3 are set before wavefront execution. Prior to start of every wavefront execution, CP/SPI sets up the register state based on enable_sgpr_* and enable_vgpr_* flags in amd_kernel_code_t object:

SGPRs before the Work-Group Ids are set by **CP** using the 16 **User Data registers**.

SGPR register numbers used for enabled registers are dense starting at SGPR0: the first enabled register is SGPR0, the next enabled register is SGPR1 etc.; disabled registers do not have an SGPR number. Because of hardware constraints, the initial SGPRs comprise up to 16 User SRGPs that are set up by CP and apply to all waves of the grid. It is possible to specify more than 16 User SGPRs using the enable_sgpr_* bit fields, in which case **only the first 16 are actually initialized**. These are then immediately followed by the System SGPRs that are set up by ADC/SPI and can have different values for each wave of the grid dispatch.

In this example, we load the buffer address from the user data **registers 2 and 3**.
```cpp
s_load_dwordx4 s[8:11], s[2:3]
```

After that, we load the per-vertex data from the buffer based on the address stored in registers 8 - 19. Which position the current thread should to loaded from the buffer is determined by the `v0`. VGPR register numbers used for enabled registers are dense starting at VGPR0: the first enabled register is VGPR0, the next enabled register is VGPR1 etc.; disabled registers do not have a VGPR number. VGPR v0 is always initialized with **a work-item ID** in the x dimension (in commpute shader). Registers v1 and v2 can be initialized with work-item IDs in the y and z dimensions, respectively.

In the vertex shader, `v0` is intialized with the **vertex index**. The number of the `buffer_load_format` instruction is 3, which means that the vertex shader has **3 vertex attributes**.

The fetch shader is not part of the vertex shader. It is stored separately. The GPU address of the fetch shader can be obtained from the user data registers.
```cpp
int32_t fetchRegIndex = vsIsaProcessor.GetUsageRegisterIndex(EShaderInputUsage::kShaderInputUsageSubPtrFetchShader);
const PtPointer fsCode = *reinterpret_cast<PtPointer const*>(&GetGpuRegs()->SPI.VS.USER_DATA[fetchRegIndex]);
```

### Vertex Buffer And Vertex Layout

The vertex buffer is the same as the constant buffer mentioned above. We can get the GPU address from the buffer descriptor, which is the same as the constant buffer. Unlike the constant buffer, the vertex buffer needs the format information, while the constant buffer doesn't need these. The buffer format is specified in the fourth word of the buffer resource descriptor. Here is the member layout of the word3.
```cpp
union SQ_BUF_RSRC_WORD3 
{
    unsigned int                       DST_SEL_X : 3;
    unsigned int                       DST_SEL_Y : 3;
    unsigned int                       DST_SEL_Z : 3;
    unsigned int                       DST_SEL_W : 3;
    unsigned int                      NUM_FORMAT : 3;
    unsigned int                     DATA_FORMAT : 4;
    unsigned int                    ELEMENT_SIZE : 2;
    unsigned int                    INDEX_STRIDE : 2;
    unsigned int                  ADD_TID_ENABLE : 1;
    unsigned int                     ATC__CI__VI : 1;
    unsigned int                     HASH_ENABLE : 1;
    unsigned int                            HEAP : 1;
    unsigned int                   MTYPE__CI__VI : 3;
    unsigned int                            TYPE : 2;
};
```
The `BUF_NUM_FORMAT` and `BUF_DATA_FORMAT` specifies data and numeric formats used by the operation. The default numeric format is `BUF_NUM_FORMAT_UNORM`. The default data format is `BUF_DATA_FORMAT_8`.

```cpp
SVertexElement vtxElement;
vtxElement.m_vertexNumFormat = PtGfx::BUF_NUM_FORMAT(vtxBuffer->m_bufferSrd.word3.bitfields.NUM_FORMAT);
vtxElement.m_vertexDataFormat = PtGfx::BUF_DATA_FORMAT(vtxBuffer->m_bufferSrd.word3.bitfields.DATA_FORMAT);
```
Supported data formats are defined in the following structure:
```cpp
typedef enum BUF_DATA_FORMAT {
BUF_DATA_FORMAT_INVALID                  = 0x00000000,
BUF_DATA_FORMAT_8                        = 0x00000001,
BUF_DATA_FORMAT_16                       = 0x00000002,
BUF_DATA_FORMAT_8_8                      = 0x00000003,
BUF_DATA_FORMAT_32                       = 0x00000004,
BUF_DATA_FORMAT_16_16                    = 0x00000005,
BUF_DATA_FORMAT_10_11_11                 = 0x00000006,
BUF_DATA_FORMAT_11_11_10                 = 0x00000007,
BUF_DATA_FORMAT_10_10_10_2               = 0x00000008,
BUF_DATA_FORMAT_2_10_10_10               = 0x00000009,
BUF_DATA_FORMAT_8_8_8_8                  = 0x0000000a,
BUF_DATA_FORMAT_32_32                    = 0x0000000b,
BUF_DATA_FORMAT_16_16_16_16              = 0x0000000c,
BUF_DATA_FORMAT_32_32_32                 = 0x0000000d,
BUF_DATA_FORMAT_32_32_32_32              = 0x0000000e,
BUF_DATA_FORMAT_RESERVED_15              = 0x0000000f,
} BUF_DATA_FORMAT;
```

Supported numeric formats are defined below:
```cpp
typedef enum BUF_NUM_FORMAT {
BUF_NUM_FORMAT_UNORM                     = 0x00000000,
BUF_NUM_FORMAT_SNORM                     = 0x00000001,
BUF_NUM_FORMAT_USCALED                   = 0x00000002,
BUF_NUM_FORMAT_SSCALED                   = 0x00000003,
BUF_NUM_FORMAT_UINT                      = 0x00000004,
BUF_NUM_FORMAT_SINT                      = 0x00000005,
BUF_NUM_FORMAT_SNORM_OGL__SI__CI         = 0x00000006,
BUF_NUM_FORMAT_RESERVED_6__VI            = 0x00000006,
BUF_NUM_FORMAT_FLOAT                     = 0x00000007,
} BUF_NUM_FORMAT;
```
After the data formats and numeric formats have been prepared, we can translate them into VKFormat. For example, `BUF_NUM_FORMAT_UNORM` and `BUF_DATA_FORMAT_8` correspond to `VK_FORMAT_R8_UNORM` format.

```cpp
VkFormat GetVkBufferFormatFromAMDFormat(PtGfx::BUF_DATA_FORMAT dataFormat, PtGfx::BUF_NUM_FORMAT numFormat)
{
	// do something
}
```
### Texture Tiling And Detile

The image resource descriptor is obtained from the user data register, which is the same as the buffer resource. 
```cpp
const uint32_t* srvDesc = reinterpret_cast<uint32_t*>(&GetGpuRegs()->SPI.PS.USER_DATA[pPixelShader->m_srvStartIndex + index * ShaderConstantDwordSize::kDwordSizeResource]);
const CTextureResourceDesc* srDesc = reinterpret_cast<const CTextureResourceDesc*>(srvDesc);
```
The naive view of an image in memory is that the pixels are stored one after another in memory usually in an X-major order. An image that is arranged in this way is called "linear". Linear images, while easy to reason about, can have very bad cache locality. Graphics operations tend to act on pixels that are close together in 2-D euclidean space. If you move one pixel to the right or left in a linear image, you only move a few bytes to one side or the other in memory. However, if you move one pixel up or down you can end up kilobytes or even megabytes away. Tiling (sometimes referred to as swizzling) is a method of re-arranging the pixels of a surface so that pixels which are close in 2-D euclidean space are likely to be close in memory.

We should convert the tiled format texture into a linear format texture before using it.

While any non-linear representation is typically referred to as "tiling", some hardware implementations actually use a more complex layout in order to provide further locality of reference. One such scheme is Morton order:
<p align="center">
    <img src="/resource/pset/image/morton.png" width="65%" height="65%">
</p>

The following code shows how to convert the linear xy index into an element index.

```cpp
static uint32_t GetElementIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t bitsPerElement, PtGfx::MicroTileMode microTileMode, PtGfx::ArrayMode arrayMode)
{
	uint32_t elem = 0;
	if (microTileMode == PtGfx::ADDR_SURF_THIN_MICRO_TILING || microTileMode == PtGfx::ADDR_SURF_DEPTH_MICRO_TILING)
	{
		elem |= ((x >> 0) & 0x1) << 0;
		elem |= ((y >> 0) & 0x1) << 1;
		elem |= ((x >> 1) & 0x1) << 2;
		elem |= ((y >> 1) & 0x1) << 3;
		elem |= ((x >> 2) & 0x1) << 4;
		elem |= ((y >> 2) & 0x1) << 5;

		return elem;
	}
	return 0;
};
```
For each pixel, we calculate the morton code offset and fetch the pixel data from the tiled image buffer. Then, we can get an image stored in linear format.

```cpp
for (int32_t z = 0; z < tiler.m_linearDepth; z++)
{
	for (int32_t y = 0; y < tiler.m_linearHeight; y++)
	{
		for (int32_t x = 0; x < tiler.m_linearWidth; x++)
		{
			uint32_t bitOffset = tiler.GetTiledElementBitOffset(x, y, z);
			uint32_t byteIndex = bitOffset / 8;
			uint8_t* src = &srcData[byteIndex];
			uint32_t destByteIndex = (z * sliceSize + y * tiler.m_linearWidth + x) * bytesPerElement;
			memcpy(&outData[destByteIndex], src, bytesPerElement);
		}
	}
}
```
### Pipeline State And Draw Index

After all resources and GPU registers have been prepared, we can finally set the pipeline state.
The depth stencil states are obtained from the `DEPTH_CONTROL` register.
The render target states are obtained from the `RENDER_TARGET` register.
The color blend state is obtained from the `CB_COLOR_CONTROL` register.

```cpp
CRHIDepthStencilState depthStencilState;
depthStencilState.bDepthTestEnable = GetGpuRegs()->DEPTH.DEPTH_CONTROL.bitfields.Z_ENABLE;
depthStencilState.bDepthWriteEnable = GetGpuRegs()->DEPTH.DEPTH_CONTROL.bitfields.Z_WRITE_ENABLE;
depthStencilState.zFunc = GetGpuRegs()->DEPTH.DEPTH_CONTROL.bitfields.ZFUNC;
```

```cpp
graphicsPsoInitDesc.m_pipelineColorBlendControl = GetGpuRegs()->CB_COLOR_CONTROL;
```

We translate these registers into the Vulkan state and set the graphics pipeline state.
```cpp
std::shared_ptr<CRHIGraphicsPipelineState> graphicsPipelineState = RHICreateGraphicsPipelineState(graphicsPsoInitDesc);
gRHICommandList.RHISetGraphicsPipelineState(graphicsPipelineState);
```

When we process the draw index PM4 packet (OP_CUS_DRAW_INDEX), we set the pipeline state, get the index buffer from the PM4 packet and execute the Vulkan draw index command.

```cpp
case OP_CUS_DRAW_INDEX:
{
	ProcessGnmPrivateOpDrawIndex(pm4Hdr, itBody);
	break;
}
```
```cpp
gRHICommandList.RHIDrawIndexedPrimitive(idxIter->second.get(), param->indexCount, 1, 0, 0, 0);
```

rendering result in loading stage:

<p align="center">
    <img src="/resource/pset/image/PS4Emulator.png" width="85%" height="85%">
</p>
---
title: How To Develop A PlayStation 4 Emulator(2)
date: 2024-07-07 21:54:02
tags:
index_img: /resource/pset/image/PS4Emulator.png
---

# Overview

# Non-Graphics Libraries

I would like to provide a brief introduction to non-graphics libraries implementation, such as libkernel, video output and system services. We will not detail it, since we mainly focus on the libraried ralated to the graphics driver.

## File Management

In order to load the game resource correctly, we map the path from the relative path to the absolute path. For the path string beginning with "/app0" or "/app1", we map it to the directory where the `eboot.bin` is located. 
```cpp
if (unMappedPath.find("/app0") != std::string::npos)
{
	retString.replace(0, sizeof("/app0"), app0Dir.c_str());
}
else if (unMappedPath.find("/app1") != std::string::npos)
{
	retString.replace(0, sizeof("/app1"), app0Dir.c_str());
}
```
## MemoryManager

PS4 has many memory protection options. For reasons of simplification, we classify them into three types: execute or read-only access, read-only access and read-only or read/write access. For CPU read protection or GPU read protection, we use the `PAGE_READONLY` protection flag. For CPU write protection option or GPU write protection option, we use `PAGE_READWRITE` protection flag. For CPU executable protection, we use the PAGE_EXECUTE_READ flag.

```cpp
uint32_t CLibKernelMemoryManager::GetProtectFlag(int prot)
{
	uint32_t protFalg = PAGE_NOACCESS;
	if ((prot & SCE_KERNEL_PROT_CPU_READ) || (prot & SCE_KERNEL_PROT_GPU_READ)){ protFalg = PAGE_READONLY; }
	if ((prot & SCE_KERNEL_PROT_CPU_WRITE) || (prot & SCE_KERNEL_PROT_GPU_WRITE)){ protFalg = PAGE_READWRITE; }
	if (prot & SCE_KERNEL_PROT_CPU_EXEC){ protFalg = PAGE_EXECUTE_READ; }
	return protFalg;
}
```
## Kernel

### Stack Smashing Protection

>Clang, GCC, and related compilers implement stack smashing protection (SSP) using StackGuard. The fundamental assumption behind the approach is that most stack buffer overflows occur by writing past the end of a function’s stack frame. In order to detect this case, a “canary value” is added to the stack before other values are declared. Before returning from the function, the stored canary value in the stack is checked. If it has been modified, a stack buffer overflow has occurred. A failure callback is invoked, which prints an error message and terminates the program. These additions are made by the compiler without your involvement. Compiler flags are used to control the degree that your program is checked. We assign a random value to __stack_chk_guard:

```cpp
static unsigned long long __stack_chk_guard = 0xdeadbeefa55a857;
```
```cpp
 { 0x7FBB8EC58F663355,"Pset___stack_chk_guard", &__stack_chk_guard },
```

winpthreads:
memory manage:
pad controller infoemation:
user service:
video output:

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

<p align="center">
    <img src="/resource/pset/image/psdriver_what_we_do.png" width="65%" height="65%">
</p>

Shader Processor Input 和 GCN ISA 关联起来
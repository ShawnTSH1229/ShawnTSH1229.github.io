---
title: How To Develop A PlayStation 4 Emulator(1)
date: 2024-07-05 23:15:36
tags:
---
# Overview
# ELF Loader & Parser

## ELF Header & SELF

In computing, the Executable and Linkable Format (ELF, formerly named Extensible Linking Format), is a common standard file format for executable files, object code, shared libraries, and core dumps. PS4 uses an extended ELF format named SELF (Signed Executable and Linkable Format) to store the executable file. Following is the SELF data layout:

<p align="center">
    <img src="/resource/pset/image/self.png" width="45%" height="45%">
</p>

Compared to ELF, SELF has two additional parts. The first part is SELF header. It describes the basic information about the SELF file. What we care about is the `Number of Segments` part. It indicates the number of the SELF segment structure followed by the SELF header.

|Offset |Size |Description |Notes|
|-------|-------|-------|-------|
| 0 | 4 | Magic | 4F 15 3D 1D|
| 0x4 | 4 | Unknown | Always 00 01 01 12|
| 0x8 | 1 | Category | 1 on SELF, 4 on PUP Entry (probably SPP). See [https://www.psdevwiki.com/ps3/Certified_File#Category PS3/PS Vita Category].|
| 0x9 | 1 | Program Type | First Half denotes version (anything between 0, oldest and F, newest), second Half denotes true type, 4 for Games, 5 for sce_module modules, 6 for Video Apps like Netflix, 8 for System/EX Apps/Executables, 9 for System/EX modules/dlls|
| 0xA | 2 | Padding ||
| 0xC | 2 | Header Size ||
| 0xE | 2 | Signature Size | ?Metadata Size?|
| 0x10 | 4 | File Size |Size of SELF|
| 0x14 | 4 | Padding ||
| 0x18 | 2 | Number of Segments | 1 Kernel, 2 SL and Secure Modules, 4 Kernel ELFs, 6 .selfs, 2 .sdll, 6 .sprx, 6 ShellCore, 6 eboot.bin, 2 sexe|
| 0x1A | 2 | Unknown | Always 0x22|
| 0x1C | 4 | Padding | |

After this, follows the segment structure. I am unable to determine the effect of the segment structure other than to guess. Due to the lack of PS4 documentation, the following content may be incorrect. The SELF segment structure is almost the same as the ELF segment. The difference may be that the data in the SELF segment structure is compressed and encrypted. These parts are used to store the exclusive segment for the PlayStation4.

```cpp
typedef struct {
 unsigned long long flags; // 0x130006 / 0x00040F / 0x000006 / 0x110006
 unsigned long long offset;
 unsigned long long encrypted_compressed_size;
 unsigned long long decrypted_decompressed_size;
} SEGMENT_TABLE;
```

The SELF program header is followed by the ELF header. Here is the ELF header information of the `eboot.bin` and the `libc.prx`.

<p align="center">
    <img src="/resource/pset/image/eboot_elf.png" width="55%" height="55%">
</p>
<p align="center">
    <img src="/resource/pset/image/libc_elf.png" width="55%" height="55%">
</p>

Here is the ELF header structure member layout. It describes the basic information of the whole ELF file, such as the object file type, target instruction set architecture, entry point address, program header table and section header table.
```cpp
typedef struct elf64_hdr {
    unsigned char	e_ident[EI_NIDENT];	/* ELF "magic number" */
    Elf64_Half e_type;
    Elf64_Half e_machine;
    Elf64_Word e_version;
    Elf64_Addr e_entry;		/* Entry point virtual address */
    Elf64_Off e_phoff;		/* Program header table file offset */
    Elf64_Off e_shoff;		/* Section header table file offset */
    Elf64_Word e_flags;
    Elf64_Half e_ehsize;
    Elf64_Half e_phentsize;
    Elf64_Half e_phnum;
    Elf64_Half e_shentsize;
    Elf64_Half e_shnum;
    Elf64_Half e_shstrndx;
} Elf64_Ehdr;
```

The `eboot.bin` shown on the above is the starup executeble file of the game. The eboot module entry point address is `99AC10`. It is a virtual address. We will allocate virtual memory and map this virtual address into virtual memory. After that, we can call the entry function to startup the program. Its ELF type is `ET_SCE_DYNEXEC`. PS4 has five exclusive ELF types:
```cpp
#define ET_SCE_EXEC	0xFE00
#define ET_SCE_REPLAY_EXEC	0xFE01
#define ET_SCE_RELEXEC	0XFE04
#define ET_SCE_STUBLIB	0xFE0C
#define ET_SCE_DYNEXEC	0xFE10
#define ET_SCE_DYNAMIC	0xFE18
```

Another ELF file is `libc.prx`. The `prx` stands for Playstation Relocatable eXecutable. Its ELF type is `ET_SCE_DYNAMIC`. We will link this module dynamically at runtime. 

We don't care about SELF information except the SELF segment structure. Those unused information in SELF is discarded. We create the ELF file and fill the segment data with the information contained in the SELF segment structure.
```cpp
m_module->m_elfData.resize(segmentsEnd);
memcpy(m_module->m_elfData.data(), elf64Header, segmentsStart);

for (uint32_t index = 0; index < numSegments; index++)
{
	if ((pSegStructureSelfs[index].m_flags & uint64_t(ESegFlags::SF_BFLG)) != 0)
	{
		uint32_t flagID = FlagsId(pSegStructureSelfs[index].m_flags);
		memcpy(m_module->m_elfData.data() + elf64ProgramHeader[flagID].p_offset, selfRawData.data() + pSegStructureSelfs[index].m_offsets, pSegStructureSelfs[index].m_encryptedCompressedSize);
	}
}
```

## Parser

### Parse Program Header
An executable or shared object file's program header table is an array of structures, each describing a segment or other information the system needs to prepare the program for execution. An object file segment comprises one or more sections, though this fact is transparent to the program header. Whether the file segment holds one or many sections also is immaterial to program loading. Nonetheless, various data must be present for program execution, dynamic linking, and so on. The order and membership of sections within a segment may vary.

program header structure:
```cpp
typedef struct {
	Elf64_Word	p_type;
	Elf64_Word	p_flags;
	Elf64_Off	p_offset;
	Elf64_Addr	p_vaddr;
	Elf64_Addr	p_paddr;
	Elf64_Xword	p_filesz;
	Elf64_Xword	p_memsz;
	Elf64_Xword	p_align;
} Elf64_Phdr;
```
Segment information is described in the program header. We iterate the program header table and obtain segment information:
```cpp
	SPtModuleInfo& moduleInfo = m_module->m_moduleInfo;
	elf64_hdr* pElfHdr = (elf64_hdr*)(m_module->m_elfData.data());
	elf64_phdr* pElfpHdr = (elf64_phdr*)(pElfHdr + 1);

	for (uint32_t index = 0; index < pElfHdr->e_phnum; index++)
	{
		switch (pElfpHdr[index].p_type)
		{
		case PT_NOTE:
		case PT_LOAD:
		case PT_SCE_RELRO:
		case PT_SCE_COMMENT:
			break;
		case PT_SCE_PROCPARAM:
			moduleInfo.m_nProcParamOffset = pElfpHdr[index].p_paddr;
			break;
		case PT_INTERP:
			moduleInfo.m_pInterProgram = m_module->m_elfData.data() + pElfpHdr[index].p_offset;
			break;
		case PT_DYNAMIC:
			moduleInfo.m_pDynamicEntry = m_module->m_elfData.data() + pElfpHdr[index].p_offset;
			moduleInfo.m_nDynamicEntryCount = pElfpHdr[index].p_filesz / sizeof(Elf64_Dyn);
			PSET_EXIT_AND_LOG_IF(moduleInfo.m_nDynamicEntryCount == 0, "empty dynamic entry");
			break;
		case PT_SCE_DYNLIBDATA:
			moduleInfo.m_pSceDynamicLib.m_pAddress = m_module->m_elfData.data() + pElfpHdr[index].p_offset;
			moduleInfo.m_pSceDynamicLib.m_size = pElfpHdr[index].p_filesz;
			break;
		default:
			PSET_LOG_INFO("unknown program header type");
		}
	}
```

We only care about four types of segment: `PT_DYNAMIC`, `PT_SCE_PROCPARAM` and `PT_SCE_DYNLIBDATA`. The first type is the standard ELF segment type. The latter two types are exclusive to the PlayStation4.

`PT_SCE_PROCPARAM`: This field stores the offset of the process parameters. We obtain the process parameters address from the offset. The process parameters comprise the size, the magic number, the entry count and the PS SDK version.
```cpp
struct SSceProcParam
{
	uint64_t m_size;
	char m_magic[4];
	uint32_t m_entryCount;
	uint64_t m_sdkVersion;
};
```
When the program executes the `sceKernelGetProcParam` function, we return the process parameters address retrieved from the segment.
```cpp
SSceProcParam* PSET_SYSV_ABI Pset_sceKernelGetProcParam(void)
{
	PSET_LOG_IMPLEMENTED("implemented function: Pset_sceKernelGetProcParam");
	SSceProcParam* retProcParam = (SSceProcParam*)GetElfModuleLoder()->GetEbootProcParam();
	assert(retProcParam != nullptr);
	return retProcParam;
}
```
`PT_DYNAMIC`: The array element specifies dynamic linking information.
`PT_SCE_DYNLIBDATA`: The dynamic library data.

### Parse Dynamic Sections

If an object file participates in dynamic linking, its program header table will have an element of type `PT_DYNAMIC`. This segment contains the `.dynamic` section. A special symbol, `_DYNAMIC`, labels the section, which contains an array of the following structures. The `_DYNAMIC` array corresponds to `m_pDynamicEntry` obtained in the last step.

```cpp
typedef struct {
	Elf64_Sxword	d_tag;
   	union {
   		Elf64_Xword	d_val;
   		Elf64_Addr	d_ptr;
	} d_un;
} Elf64_Dyn;

```
For each object with this type, `d_tag` controls the interpretation of `d_un`.
`d_val`:  These objects represent integer values with various interpretations.
`d_ptr`:  It stores the relative address to the base address. The base address is stored in 'm_pSceDynamicLib', which is the segment of `PT_SCE_DYNLIBDATA` type.

We divided the dynamic sections into two different types. The first type is segments that don't require access to string tables. The string table (`DT_SCE_STRTAB`) is a section that contains strings representing the file name, the module name and the library name. Since some dynamic sections depend on the string stable, we access these sections after all of the first types of the section iteration have been finished.

First Iteration:
```cpp
SPtModuleInfo& moduleInfo = m_module->m_moduleInfo;
uint8_t* baseAddress = (uint8_t*)moduleInfo.m_pSceDynamicLib.m_pAddress;
for (uint32_t index = 0; index < moduleInfo.m_nDynamicEntryCount; index++)
{
	Elf64_Dyn& elf64Dyn = ((Elf64_Dyn*)moduleInfo.m_pDynamicEntry)[index];
	switch (elf64Dyn.d_tag)
	{
	case DT_INIT:
		moduleInfo.m_nInitOffset = elf64Dyn.d_un.d_ptr;
		break;
	case DT_SCE_STRTAB:
		moduleInfo.m_sceStrTable.m_pAddress = baseAddress + elf64Dyn.d_un.d_ptr;
	case DT_SCE_STRSZ:
		moduleInfo.m_sceStrTable.m_size = elf64Dyn.d_un.d_val;
		break;
	case DT_SCE_RELA:
		moduleInfo.m_relocationTable.m_pAddress = baseAddress + elf64Dyn.d_un.d_ptr;;
		break;
	case DT_SCE_RELASZ:
		moduleInfo.m_relocationTable.m_size = elf64Dyn.d_un.d_val / sizeof(Elf64_Rela);
		break;
	case DT_SCE_JMPREL:
		moduleInfo.m_relocationPltTable.m_pAddress = baseAddress + elf64Dyn.d_un.d_ptr;;
		break;
	case DT_SCE_PLTRELSZ:
		moduleInfo.m_relocationPltTable.m_size = elf64Dyn.d_un.d_val / sizeof(Elf64_Rela);
		break;
	case DT_SCE_SYMTAB:
		moduleInfo.m_symbleTable.m_pAddress = baseAddress + elf64Dyn.d_un.d_ptr;
		break;
	case DT_SCE_SYMTABSZ:
		moduleInfo.m_symbleTable.m_size = elf64Dyn.d_un.d_val;
		break;
	}
}
```
We iterate over the sections that don't require string tables at first. It contains the following types of sections:

`DT_INIT`: It stores the initialization function offset. We can get the initialization function address from the function offset:
```cpp
moduleInfo.m_pInitProc = moduleInfo.m_nInitOffset + (uint8_t*)moduleInfo.m_mappedMemory.m_pAddress;
```
The initialization function is executed after all modules have been loaded.
```cpp
void CPtModuleLoader::InitNativeModules()
{
	for (uint32_t index = 0; index < m_nNativeModules; index++)
	{
		if (m_natveModules[index].m_fileName != "eboot.bin")
		{
			auto initFunc = reinterpret_cast<InitProc>(m_natveModules[index].m_moduleInfo.m_pInitProc);
			initFunc(0, nullptr, nullptr);
		}
	}
}
```
`DT_SCE_STRTAB` `DT_SCE_STRSZ`: This element holds the string table address and size, as described above. It is a type of section that is exclusive to PS4.

`DT_SCE_RELA`: This element holds the address of a relocation table. Entries in the table have explicit addends, such as `Elf32_Rela` for the 32-bit file class or `Elf64_Rela` for the 64-bit file class. An object file may have multiple relocation sections. When building the relocation table for an executable or shared object file, the link editor catenates those sections to form a single table. Although the sections remain independent in the object file, the dynamic linker sees a single table. When the dynamic linker creates the process image for an executable file or adds a shared object to the process image, it reads the relocation table and performs the associated actions.

`DT_SCE_RELASZ`: This element holds the total size, in bytes, of the `DT_SCE_RELA` relocation table.

`DT_SCE_JMPREL` `DT_SCE_PLTRELSZ`: Lazy binding (also known as lazy linking or on-demand symbol resolution) is the process by which symbol resolution isn't done until a symbol is actually used. Functions can be bound on-demand, but data references can't. All dynamically resolved functions are called via a Procedure Linkage Table (PLT) stub. A PLT stub uses relative addressing, using the Global Offset Table (GOT) to retrieve the offset. The PLT knows where the GOT is, and uses the offset to this table (determined at program linking time) to read the destination function's address and make a jump to it. **In PSET4, we treat the PLT the same as the relocation table and don't perform lazy binding.**

`DT_SCE_SYMTAB` `DT_SCE_SYMTABSZ`: This element holds the address of the symbol table, with `Elf32_Sym` entries for the 32-bit class of files and `Elf64_Sym` entries for the 64-bit class of files.

Second Iteration:
```cpp
	uint8_t* sceStrTable = (uint8_t*)moduleInfo.m_sceStrTable.m_pAddress;
	for (uint32_t index = 0; index < moduleInfo.m_nDynamicEntryCount; index++)
	{
		Elf64_Dyn& elf64Dyn = ((Elf64_Dyn*)moduleInfo.m_pDynamicEntry)[index];
		switch (elf64Dyn.d_tag)
		{ 
		case DT_NEEDED:
		{
			char* fileName = (char*)&sceStrTable[elf64Dyn.d_un.d_ptr];
			m_module->m_neededFiles.push_back(fileName);

			PSET_LOG_INFO(std::string("DT_NEEDED:") + fileName);
			break;
		}

		case DT_SCE_MODULE_INFO:
		case DT_SCE_NEEDED_MODULE:
		{
			SSPtModuleValue moduleValue;
			moduleValue.value = elf64Dyn.d_un.d_val;
			m_module->m_id2ModuleNameMap.emplace(std::make_pair(moduleValue.id, (const char*)(&sceStrTable[moduleValue.name_offset])));
			break;
		}

		case DT_SCE_IMPORT_LIB:
		case DT_SCE_EXPORT_LIB:
		{
			SPtLibraryValue libValue;
			libValue.m_value = elf64Dyn.d_un.d_val;
			m_module->m_id2LibraryNameMap.emplace(std::make_pair(libValue.id, (const char*)(&sceStrTable[libValue.name_offset])));
			break;
		}

		case DT_SCE_ORIGINAL_FILENAME:
		{
			SPtLibraryValue libValue;
			libValue.m_value = elf64Dyn.d_un.d_val;
			PSET_LOG_INFO(std::string("DT_SCE_ORIGINAL_FILENAME:") + (char*)&sceStrTable[libValue.name_offset]);
			break;
		}
		}
	}
```
After that, we iterate over the sections that require the string table. It contains the following types of sections:

`DT_NEEDED`: This element holds the string table offset of a null-terminated string, giving the name of a needed library. The offset is an index into the table recorded in the `DT_SCE_STRTAB` code. The dynamic array may contain multiple entries with this type. These entries' relative order is significant, though their relation to entries of other types is not. Following is the needed libraries of the `eBoot` module:

<p align="center">
    <img src="/resource/pset/image/DT_NEEDED.png" width="55%" height="55%">
</p>

`DT_SCE_MODULE_INFO` `DT_SCE_NEEDED_MODULE`: These elements store the current module information and the needed module information. Each module contains one or more libraries. Here is the module information of the `eBoot` module:
<p align="center">
    <img src="/resource/pset/image/DT_MODULE.png" width="55%" height="55%">
</p>

`DT_SCE_IMPORT_LIB` `DT_SCE_EXPORT_LIB`: Those elements hold the import and export libraries. It will be used in the latter dynamic link process.
<p align="center">
    <img src="/resource/pset/image/DT_LIB.png" width="55%" height="55%">
</p>

## Load ELF Module Into Memory

As we have all segment information, we can load them into memory. Only headers with a type of PT_LOAD or PT_SCE_RELRO describe a loadable segment.

```cpp
for (auto const& phdr : m_module->m_aSegmentHeaders)
{
	switch (phdr.p_type)
	{
		case PT_LOAD:
		{
			if (phdr.p_flags & PF_X)
			{
				MapSegment(phdr);
			}
			else if (phdr.p_flags & PF_W)
			{
				MapSegment(phdr);
			}
			break;
		}
		case PT_SCE_RELRO:
		{
			MapSegment(phdr);
			break;
		}
	}
}
```

Load each of the loadable segments. This is performed as follows:

1.Allocate virtual memory for each segment, at the address specified by the p_vaddr member in the program header. The size of the segment in memory is specified by the `p_memsz` member.
```cpp
mappedMemory.m_pAddress = (void*)0x000000000;
if (m_module->m_fileName == "eboot.bin")
{
	mappedMemory.m_pAddress = (void*)0x800000000;
}
mappedMemory.m_pAddress = VirtualAlloc(mappedMemory.m_pAddress, mappedMemory.m_size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
```
2.Copy the segment data from the file offset specified by the p_offset member to the virtual memory address specified by the `p_vaddr` member. The size of the segment in the file is contained in the p_filesz member. This can be zero.

3.The `p_memsz` member specifies the size the segment occupies in memory. This can be zero. If the p_filesz and `p_memsz` members differ, this indicates that the segment is padded with zeros. All bytes in memory between the ending offset of the file size, and the segment's virtual memory size are to be cleared with zeros.

```cpp
void CPtElfProcessor::MapSegment(Elf64_Phdr const& hdr)
{
	SPtModuleInfo& moduleInfo = m_module->m_moduleInfo;
	uint8_t* pDstAddr = reinterpret_cast<uint8_t*>(AlignUp(size_t(moduleInfo.m_mappedMemory.m_pAddress) + hdr.p_vaddr, hdr.p_align));
	uint8_t* pSrcData = m_module->m_elfData.data() + hdr.p_offset;
	memcpy(pDstAddr, pSrcData, hdr.p_filesz);
}
```
# Dynamic Link

Dynamic linking defers much of the linking process until a program starts running. We have loaded the segment into memory. However, the memory address of the symbols using dynamic link have not been determined. We shall perform dynamic link for those symbols. These symbols are divided into two types.

The symbols of the first type are not visible outside the object file containing their definition. We can relocate after the module is loaded. Other symbols of the second type are visible to all object files being combined. They are relocated after all module loaded.

## Local Symbols Relocation

The following notations are used for specifying relocations in table:
`A`: Represents the addend used to compute the value of the relocatable field.
`B`: Represents the base address at which a shared object has been loaded into memory during execution. Generally, a shared object is built with a 0 base virtual address, but the execution address will be different.
`G`: Represents the offset into the global offset table at which the relocation entry's symbol will reside during execution.
`GOT`: Represents the address of the global offset table.
`L`: Represents the place (section offset or address) of the Procedure Linkage Table entry for a symbol.
`P`: Represents the place (section offset or address) of the storage unit being relocated (computed using `r_offset`).
`S`:  Represents the value of the symbol whose index resides in the relocation entry.

The AMD64 ABI architectures uses only `Elf64_Rela` relocation entries with explicit addends. The `r_addend` member serves as the relocation addend.
Table: Relocation Types
|Name	|Value	|Field	|Calculation
|-------|-------|-------|-------|
|R_X86_64_NONE	|0	|none	|none|
|R_X86_64_64	|1	|word64	|S + A|
|R_X86_64_PC32	|2	|word32	|S + A - P|
|R_X86_64_GOT32	|3	|word32	|G + A|
|R_X86_64_PLT32	|4	|word32	|L + A - P|
|R_X86_64_COPY	|5	|none	|none|
|R_X86_64_GLOB_DAT	|6	|word64	|S|
|R_X86_64_JUMP_SLOT	|7	|word64	|S|
|R_X86_64_RELATIVE	|8	|word64	|B + A|
|R_X86_64_GOTPCREL	|9	|word32	|G + GOT + A - P|
|R_X86_64_32	|10	|word32	|S + A|
|R_X86_64_32S	|11	|word32	|S + A|
|R_X86_64_16	|12	|word16	|S + A|
|R_X86_64_PC16	|13	|word16	|S + A - P|
|R_X86_64_8	|14	|word8	|S + A|
|R_X86_64_PC8	|15	|word8	|S + A - P|
|R_X86_64_DPTMOD64	|16	|word64	 ||
|R_X86_64_DTPOFF64	|17	|word64	 ||
|R_X86_64_TPOFF64	|18	|word64	 ||
|R_X86_64_TLSGD	|19	|word32	 ||
|R_X86_64_TLSLD	|20	|word32	 ||
|R_X86_64_DTPOFF32	|21	|word32	 ||
|R_X86_64_GOTTPOFF	|22	|word32	 ||
|R_X86_64_TPOFF32	|23	|word32	 ||

Local symbols are not visible outside the object file containing their definition. Local symbols of the same name can exist in multiple files without interfering with each other. Currently, we only process the symbols whose binding type is R_X86_64_RELATIVE. The symbol address in memory for R_X86_64_RELATIVE type is the combination of the base memory address and the addend offset.

```cpp
	SPtModuleInfo& moduleInfo = m_module->m_moduleInfo;
	uint8_t* pCodeAddress = (uint8_t*)moduleInfo.m_mappedMemory.m_pAddress;
	Elf64_Sym* pSymTable = (Elf64_Sym*)moduleInfo.m_symbleTable.m_pAddress;

	for (uint32_t index = 0; index < relaCount; index++)
	{
		Elf64_Rela* pRela = &pReallocateTable[index];
		
		const uint64_t nSymIdx = ELF64_R_SYM(pRela->r_info);
		const uint64_t bindType = ELF64_R_TYPE(pRela->r_info);

		Elf64_Sym& symbol = pSymTable[nSymIdx];
		
		uint64_t symAddend = pRela->r_addend;
		int nBinding = ELF64_ST_BIND(symbol.st_info);
		if (nBinding == STB_LOCAL)
		{
			void* symAddress = nullptr;
			switch (bindType)
			{
			case R_X86_64_RELATIVE:
			{
				symAddress = pCodeAddress + symAddend;
				break;
			}
			default:
			{
				//todo:
			}
			}
			
			*(uint64_t*)(pCodeAddress + pRela->r_offset) = reinterpret_cast<uint64_t>(symAddress);
		}
	}
```
As we mentioned above, we don't use lazy binding, which means that the PLT relocation table is the same as the common relocation table.
```cpp
SPtModuleInfo& moduleInfo = m_module->m_moduleInfo;
RelocateNativeLocalSymbol((Elf64_Rela*)moduleInfo.m_relocationTable.m_pAddress, moduleInfo.m_relocationTable.m_size);
RelocateNativeLocalSymbol((Elf64_Rela*)moduleInfo.m_relocationPltTable.m_pAddress, moduleInfo.m_relocationPltTable.m_size);
```

## Export Symbols

For those symbols are visible to all object files being combined, we shall export them and store the relocated symbol memory address into a global address table. There are two types of symbols in them: 'STB_GLOBAL' and 'STB_WEAK'.

`STB_GLOBAL`: Global symbols. These symbols are visible to all object files being combined. One file's definition of a global symbol will satisfy another file's undefined reference to the same global symbol.

`STB_WEAK`: Weak symbols. These symbols resemble global symbols, but their definitions have lower precedence.

Each symbol in PS4 has a hash value. We can get the module ID, library ID and NID by decoding the hash value. Then, the module name is retrieved from the global module name map by module ID.

```cpp
uint8_t* pCodeAddress = (uint8_t*)moduleInfo.m_mappedMemory.m_pAddress;
uint8_t* pStrTable = (uint8_t*)moduleInfo.m_sceStrTable.m_pAddress;
Elf64_Sym* pSymTable = (Elf64_Sym*)moduleInfo.m_symbleTable.m_pAddress;

uint32_t nSymbolNum = moduleInfo.m_symbleTable.m_size/ sizeof(Elf64_Sym);

for (uint32_t i = 0; i < nSymbolNum; i++)
{
	const Elf64_Sym& symbol = pSymTable[i];
	auto binding = ELF64_ST_BIND(symbol.st_info);
	char* symName = (char*)(&pStrTable[symbol.st_name]);

	if ((binding == STB_GLOBAL || binding == STB_WEAK) && symbol.st_shndx != SHN_UNDEF)
	{
		uint16_t moduleId = 0;
		uint16_t libId = 0;
		uint64_t nid = 0;
		CPtDynamicLinker::DecodeSymbol(symName, moduleId, libId, nid);
		std::string moduleName;
		std::string libName;

		auto iterModule = m_module->m_id2ModuleNameMap.find(moduleId);
		if (iterModule != m_module->m_id2ModuleNameMap.end())
		{
			moduleName = iterModule->second;
		}
		else
		{
			assert(false);
		}

		auto iterLib = m_module->m_id2LibraryNameMap.find(libId);
		if (iterLib != m_module->m_id2LibraryNameMap.end())
		{
			libName = iterLib->second;
		}
		else
		{
			assert(false);
		}
		GetPtDynamicLinker()->AddNativeModule(moduleName);
		GetPtDynamicLinker()->AddSymbol(moduleName, libName, nid, pCodeAddress + symbol.st_value, nullptr);
	}
}
```
## PS4 Shared Library

Some shared libraries are exclusive to PlayStation4, such as the libSceGnmDriver library and the libSceAudioOut library. Those libraries can't be loaded and parsed from the game's source code, since they are a built-in library in PlayStation4 and there is no need to pack them into the game source code. 

However, if a library uses a symbol implemented in the libSceGnmDriver library, where can we find and load the source code on the Windows platform? In addition, where is the symbol's address in memory? The solution is, write an override library for those PS4 built-in libraries.

The first step is to collect and generate all symbols from the PS4 built-in libraries. Symbol information can be found [<u>**here**</u>](https://github.com/idc/ps4libdoc). This repository stores built-in symbol information in JSON format. Here is an example:
```json
﻿{
  "shared_object_name": "libSceGnmDriver.prx",
  "shared_object_names": [
    "libkernel.prx",
    "libSceLibcInternal.prx",
    "libSceVideoOut.prx"
  ],
  "modules": [
    {
      "name": "libSceGnmDriver",
      "version_major": 1,
      "version_minor": 1,
      "libraries": [
        {
          "name": "libSceGnmDebugModuleReset",
          "version": 1,
          "is_export": true,
          "symbols": [
            {
              "id": 8548889540294976816,
              "hex_id": "76A3C1BE3155A530",
              "encoded_id": "dqPBvjFVpTA",
              "name": "sceGnmDebugModuleReset"
            }
          ]
        },
        {
          "name": "libSceGnmDebugReset",
          "version": 1,
          "is_export": true,
          "symbols": [
            {
              "id": 4959518870559534216,
              "hex_id": "44D3C022D88C2C88",
              "encoded_id": "RNPAItiMLIg",
              "name": "sceGnmDebugReset"
            },
            {
              "id": 14178220821813346673,
              "hex_id": "C4C328B7CF3B4171",
              "encoded_id": "xMMot887QXE",
              "name": null
            }
          ]
        }]
    }
  ]
}
```

`is_export` indicates if that library is exported, if false, it is imported.

* `type` when not present is `Function`. Can be `Function`, `Object`, `TLS`, or `Unknown11` (TBD).
* `name` is either not present or is `null` when the name for the symbol is unknown.
* `hex_id` and `encoded_id` are included for human convenience and are not used by tools.

We developed an automation tool to convert the PS4 symbol's JSON file into C++ source code. It is located at `tools\SceModuleGenerator`. Here is the part of the `sceGnmDriver` header after converting. All generated symbols are added a prefix named `Pset`.
```cpp
#pragma once
#include "overridemodule/PsetLibraryCommon.h" 

int PSET_SYSV_ABI Pset_sceGnmAddEqEvent(void);
int PSET_SYSV_ABI Pset_sceGnmAreSubmitsAllowed(void);
int PSET_SYSV_ABI Pset_sceGnmBeginWorkload(void);
int PSET_SYSV_ABI Pset_sceGnmComputeWaitOnAddress(void);
int PSET_SYSV_ABI Pset_sceGnmComputeWaitSemaphore(void);
int PSET_SYSV_ABI Pset_sceGnmCreateWorkloadStream(void);
int PSET_SYSV_ABI Pset_sceGnmDebuggerGetAddressWatch(void);
int PSET_SYSV_ABI Pset_sceGnmDebuggerHaltWavefront(void);
//......
```
Following is the `sceGnmDriver` source code. Currently, these functions will do nothing when invoked except output the log since we haven't implemented them.
```cpp
//Generated By SceModuleGenerator
#include "pset_libSceGnmDriver.h"

int PSET_SYSV_ABI Pset_sceGnmAddEqEvent(void)
{
	PSET_LOG_UNIMPLEMENTED("unimplemented function: Pset_sceGnmAddEqEvent");
	return PSET_OK;
}

int PSET_SYSV_ABI Pset_sceGnmAreSubmitsAllowed(void)
{
	PSET_LOG_UNIMPLEMENTED("unimplemented function: Pset_sceGnmAreSubmitsAllowed");
	return PSET_OK;
}

//......
```

The next step is to export these symbols into the global symbol address table. Each module comprises one or more libraries. Each library contains a set of function entries.
```cpp
struct SPSET_LIB_EXPORT_SYSMBOL
{
	const uint64_t m_nid;
	const char* m_funcName;
	const void* m_pFunction;
};
#define SPSET_LIB_EXPORT_FUNTCTION_END {0,nullptr,nullptr}

struct SPSET_EXPORT_LIB
{
	const char* m_libName;
	const SPSET_LIB_EXPORT_SYSMBOL* m_pFunctionEntries;
};
#define SPSET_EXPORT_LIB_END {nullptr,nullptr}

struct SPSET_EXPORT_MODULE
{
	SPSET_EXPORT_MODULE(const char* moduleName, const SPSET_EXPORT_LIB* pLibraries);
	const char* m_moduleName;
	const SPSET_EXPORT_LIB* m_pLibraries;
};
```
The symbol table is generated automatically using `SceModuleGenerator`:
```cpp
static const SPSET_LIB_EXPORT_SYSMBOL gSymTable_libSceGnmDriver_libSceGnmDriver[] =
{
 { 0x6F4C729659D563F2,"Pset_sceGnmAddEqEvent", (void*)Pset_sceGnmAddEqEvent },
 { 0x6F4F0082D3E51CF8,"Pset_sceGnmAreSubmitsAllowed", (void*)Pset_sceGnmAreSubmitsAllowed },
 { 0x8A1C6B6ECA122967,"Pset_sceGnmBeginWorkload", (void*)Pset_sceGnmBeginWorkload },
 { 0x7DFACD40EB21A30B,"Pset_sceGnmComputeWaitOnAddress", (void*)Pset_sceGnmComputeWaitOnAddress },
 { 0x1096A9365DBEA605,"Pset_sceGnmComputeWaitSemaphore", (void*)Pset_sceGnmComputeWaitSemaphore },
 { 0xE6E7409BEE9BA158,"Pset_sceGnmCreateWorkloadStream", (void*)Pset_sceGnmCreateWorkloadStream },
}
```
It contains three parts:
`m_nid`: The Nid is an unique value for each symbol. It's the key of the symbol in global symbol map. The symbol address is retrieved by the Nid during the relocation process.
`m_funcName`: Symbol name.
`m_pFunction`: Symbol address.

```cpp
static const SPSET_EXPORT_LIB gLibTable_libSceGnmDriver[] =
{
 {"libSceGnmDriver", gSymTable_libSceGnmDriver_libSceGnmDriver },
 {"libSceGnmDebugModuleReset", gSymTable_libSceGnmDebugModuleReset_libSceGnmDriver },
 {"libSceGnmDebugReset", gSymTable_libSceGnmDebugReset_libSceGnmDriver },
 {"libSceGnmDriverCompat", gSymTable_libSceGnmDriverCompat_libSceGnmDriver },
 {"libSceGnmDriverResourceRegistration", gSymTable_libSceGnmDriverResourceRegistration_libSceGnmDriver },
 {"libSceGnmWaitFreeSubmit", gSymTable_libSceGnmWaitFreeSubmit_libSceGnmDriver },
 SPSET_EXPORT_LIB_END
};
```
Dynamic linker exports the module **before** loading and parsing the ELF file. The modules are exported into the global symbol address table.
```cpp
static const SPSET_EXPORT_MODULE gExportModule_libSceGnmDriver("libSceGnmDriver", gLibTable_libSceGnmDriver);
```
```cpp
SPSET_EXPORT_MODULE::SPSET_EXPORT_MODULE(const char* moduleName, const SPSET_EXPORT_LIB* pLibraries)
{
	GetPtDynamicLinker()->InitializeOverrideModule(moduleName, pLibraries);
}
```
## Relocate Global Symbols

All symbols have been loaded. The symbols from the game library are loaded during ELF parsing. PS4 built-in library symbols are loaded during emulator initialization. The next step is to relocate the symbols retrieved from the game ELF file and replaced it with the address in the memory.
```cpp
	SPtModuleInfo& moduleInfo = nativeModule.GetModuleInfo();
	uint8_t* pCodeAddress = (uint8_t*)moduleInfo.m_mappedMemory.m_pAddress;
	uint8_t* pStrTable = (uint8_t*)moduleInfo.m_sceStrTable.m_pAddress;
	Elf64_Sym* pSymTable = (Elf64_Sym*)moduleInfo.m_symbleTable.m_pAddress;

	for (uint32_t index = 0; index < relaCount; index++)
	{
		Elf64_Rela* pRela = &pReallocateTable[index];
		const uint64_t nSymIdx = ELF64_R_SYM(pRela->r_info);

		Elf64_Sym& symbol = pSymTable[nSymIdx];
		auto nBinding = ELF64_ST_BIND(symbol.st_info);
		const char* pEncodedName = (const char*)&pStrTable[symbol.st_name];

		if (nBinding == STB_GLOBAL || nBinding == STB_WEAK)
		{
			if (!IsEncodedSymbol(pEncodedName))
			{
				continue;
			}
			uint64_t decodedNid = 0;
			uint16_t decodedModuleId = 0;
			uint16_t decodedLibId = 0;
			DecodeSymbol(pEncodedName, decodedModuleId, decodedLibId, decodedNid);

			std::string libName;
			std::string moduleName;
			nativeModule.GetLibAndModuleName(decodedLibId, decodedModuleId, libName, moduleName);

			std::string reDirlibName = libName;
			std::string reDirmoduleName = moduleName;

			const void* symAddress = GetSymbolAddress(reDirmoduleName, reDirlibName, decodedNid);
			if (symAddress)
			{
				*(uint64_t*)(pCodeAddress + pRela->r_offset) = reinterpret_cast<uint64_t>(symAddress);
			}
		}
	}
```
It contains the following steps:

1.Obtain the Nid, module ID and library ID by decoding the symbol name.
```cpp
DecodeSymbol(pEncodedName, decodedModuleId, decodedLibId, decodedNid);
```
2.Retrieve the library name and the module name by module ID and library ID.
```cpp
nativeModule.GetLibAndModuleName(decodedLibId, decodedModuleId, libName, moduleName);
```
3.Get the symbol address in memory by the Nid, module name and library name
```cpp
const void* symAddress = GetSymbolAddress(reDirmoduleName, reDirlibName, decodedNid);
```
4.Relocate the symbol address
```cpp
*(uint64_t*)(pCodeAddress + pRela->r_offset) = reinterpret_cast<uint64_t>(symAddress);
```
# Run The Emulator

We are now able to run the emulator once the program has been loaded. Create a new thread to run the game program.
```cpp
pthread_t retTid;
pthread_attr_t attr;
pthread_attr_init(&attr);
pthread_attr_setstacksize(&attr, defaultStackSize); //TODO: get stack size from program parameters
int retRes = pthread_create(&retTid, &attr, CPsetThread::RunThreadFunction, this);
PSET_EXIT_AND_LOG_IF(retRes != 0, "pthread create faild");
pthread_attr_destroy(&attr);
pthread_join(retTid, nullptr);
```

The Executable and Linkable Format is standardized as an adaptable file format in the System V ABI. We shall invoke the main function in sysv_abi's calling convention.
```cpp
void* PSET_SYSV_ABI CPsetThread::SysABIMainFuncttion(void* args)
{
	CPsetThread* psetThread = (CPsetThread*)args;
	((PsetMainFunction)psetThread->m_pMainFunction)(&psetThread->m_psetMainArg, CPsetThread::DefaultExitFunction);
	return nullptr;
}
```

The System V Application Binary Interface is a set of specifications that detail calling conventions, object file formats, executable file formats, dynamic linking semantics, and much more for systems that complies with the X/Open Common Application Environment Specification and the System V Interface Definition. It is today the standard ABI used by the major Unix operating systems such as Linux, the BSD systems, and many others. The Executable and Linkable Format (ELF) is part of the System V ABI.

The entry function needs an additional parameter that indicates the startup module name:
```cpp
struct SPsetMainArgs
{
	uint64_t    m_argc = 1;
	const char* m_argv[1] = { "eboot.bin" };
};
```
Although the emulator compiles and runs successfully, we didn't get the expected results:

<p align="center">
    <img src="/resource/pset/image/unimplemented_funcs.png" width="85%" height="85%">
</p>

All output logs are "unimplemented functions: xxxxxx", since we haven't implemented them. In the [<u>**next blog**</u>](https://github.com/idc/ps4libdoc), we will implement these functions in the PS4 built-in shared library.  

[<u>**TODO:TODO:PS4 Emulator Source Code**</u>](https://github.com/ShawnTSH1229/VkRtInCryEngine)







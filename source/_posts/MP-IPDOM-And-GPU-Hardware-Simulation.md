---
title: MP_IPDOM And GPU Hardware Simulation
date: 2024-10-03 20:00:18
tags:
---

## Overview
Based on thie post

## GPU Brach Divergence

## Multi-Path IPDOM

SIMT stack is a single-path stack execution model. Its serialization of divergent control flow paths reduces thread-level parallelism. MP IPDOM enables multi-path execution with support for an arbitrary number of diverged splits while maintaining reconvergence at the IPDOM. 

MP IPDOM observes that the SIMT stack items could be divided into two parts based on their role: execution and reconvergence waiting.The core idea is that those items for execution could be executed parallelly, which means the warp splits generated after divergence could be executed individually in a separate control path.

To achieve this we replace the SIMT reconvergence stack structure with two tables. 

The warp Split Table (ST) records the state of warp splits executing in parallel basic blocks (i.e., blocks that do not dominate each other), which can therefore be scheduled concurrently. 

Please note that we reused the simt structure as the split table. The only difference is the R-Index, which would be described later.

```cpp
  struct simt_stack_entry 
  {
    address_type m_pc;
    unsigned int m_calldepth;
    simt_mask_t m_active_mask;
    address_type m_recvg_pc;
    unsigned long long m_branch_div_cycle;
    
    //GPGPULearning:ZSY_MPIPDOM:[BEGIN]
    int m_r_index;
    //GPGPULearning:ZSY_MPIPDOM:[END]

    stack_entry_type m_type;
  }
```


The Reconvergence Table (RT) records reconvergence points for the splits. 
```cpp
  struct simt_recovergence_table
  {
      address_type m_pc;
      address_type m_recvg_pc;
      simt_mask_t m_recvg_mask;
      simt_mask_t m_pending_mask;
      unsigned m_split_wid;

      struct st_extra_info
      {
          unsigned int m_calldepth;
          unsigned long long m_branch_div_cycle;
          stack_entry_type m_type;
      };

      st_extra_info extro_info;
  };
```

The ST and RT tables work cooperatively to ensure splits executing parallel basic blocks will reconverge at IPDOM points.

### The Basic Workflow of MP IPDOM

**TODO PRC TABLE !!!!!!!!!!!** 

We will use a simple CUDA program to describe this part. The control flow graph is the same as the example shown in the original paper.

```cpp
__global__ void test_mpipdom_dualpath(int *data_in, int *data_out, int numElements) {
  int tid = threadIdx.x;
  if (tid % 2 == 0) {
    data_out[tid] = data_in[tid] + 10000;
  } else {
    data_out[tid] = data_in[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
    data_out[tid] = data_out[tid] + 100;
  }
  data_out[tid] += 1;
}
```
<p align="center">
    <img src="/resource/mpipdom/image/simple_example.png" width="60%" height="60%">
</p>

Below shows the operation of the MP IPDOM illustrating changes to the ST and RT tables.

<p align="center">
    <img src="/resource/mpipdom/image/mpipdom_operation.png" width="60%" height="60%">
</p>

Step1: The warp begins executing at block A. Since there is no divergence, there is only a single entry in the ST, and the RT is empty.

Step2A: Then, the warp is scheduled on the pipeline until it reaches the end of block A. After the warp executes branch  {% katex %}BR^{A}_{B−C}{% endkatex %}on cycle 2, warp A1111 diverges into two splits B0101 and C1010. Then, the A1111 entry is moved from the ST to the RT.

If divergence occurs and the new recovergence PC is not equal to the last recovergence PC, add a new recovergence item to the recovergence table.

```cpp
if ((num_divergent_paths > 1) && !warp_diverged) 
{
    warp_diverged = true;
    new_recvg_pc = recvg_pc;
    if (new_recvg_pc != top_recvg_pc)
    {
        real_rt_table.push_back(simt_recovergence_table());
        ......
    }
}
```

Step2B(1):with PC field set to the RPC (recovergence PC) of branch {% katex %}BR^{A}_{B−C}{% endkatex %} (i.e., D).

```cpp
real_rt_table.back().m_pc = new_recvg_pc;
```
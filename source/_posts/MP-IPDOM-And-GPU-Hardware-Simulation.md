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

We will use a simple CUDA program to describe this part. The control flow graph is similar to the example shown in the original paper.

```cpp
__global__ void test_mpipdom_dualpath(int *data_in, int *data_out, int numElements) {
  int tid = threadIdx.x;
  if (tid < 8) {
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

Please note that we should disable NVCC compile optimization when we compile the cuda program.

```cpp
nvcc --cudart shared -O 0 -g -Xcicc -O0 -Xptxas -O0 -o test_cuda test_cuda.cu
```

Below shows the operation of the MP IPDOM illustrating changes to the ST and RT tables.

<p align="center">
    <img src="/resource/mpipdom/image/mpipdom_operation.png" width="60%" height="60%">
</p>

Step1: 
>The warp begins executing at block A. Since there is no divergence, there is only a single entry in the ST, and the RT is empty.

Step2A: 
>Then, the warp is scheduled on the pipeline until it reaches the end of block A. After the warp executes branch  {% katex %}BR^{A}_{B−C}{% endkatex %}on cycle 2, warp A1111 diverges into two splits B0101 and C1010. Then, the A1111 entry is moved from the ST to the RT.

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

The log shows that divergence occurs at cycle 21. The first divergent path contains threads from 8 to 31 in the warp, corresponding to FFFFFF00. Another divergent path contains threads from 8 to 7 in the warp, corresponding to 000000FF.

<p align="center">
    <img src="/resource/mpipdom/image/warp_split.png" width="60%" height="60%">
</p>


Step2B(1):
>with PC field set to the RPC (recovergence PC) of branch {% katex %}BR^{A}_{B−C}{% endkatex %} (i.e., D).

```cpp
real_rt_table.back().m_pc = new_recvg_pc;
```
In our example, the start PCs of branches B and C are 48 and 136 respectively. They are recovergence at PC 424, which is shown below.

<p align="center">
    <img src="/resource/mpipdom/image/new_item_pc.png" width="75%" height="75%">
</p>

Step2B(2):
>The RPC can be determined at compile time and either conveyed using an additional instruction before the branch or encoded as part of the branch itself (current GPUs typically include additional instructions to manipulate the stack of active masks).

The reconvergence PC (RPC) of the new RT (reconvergence table) item is equal to the RPC of the last ST (split table) item. In this example, it equals -1. We will describe it more deeply in the next section that introduces nested MP IPDOM.

```cpp
real_rt_table.back().m_recvg_pc = m_stack.back().m_recvg_pc;
```

Step2B(3):
>The Reconvergence Mask entry is set to the same value of the active mask of the diverged warp split before the branch.
```cpp
real_rt_table.back().m_recvg_mask =  m_stack.back().m_active_mask;
```

Before the branches B and C, there is no divergence, which means the active mask is equal to FFFFFFFF.
<p align="center">
    <img src="/resource/mpipdom/image/recon_mask.png" width="60%" height="60%">
</p>

Step2B(4):
>The Pending Mask entry is used to represent threads that have not yet reached the reconvergence point. Hence, it is also initially set to the same value as the active mask

```cpp
real_rt_table.back().m_pending_mask =  m_stack.back().m_active_mask;
```
<p align="center">
    <img src="/resource/mpipdom/image/pend_mask_init.png" width="60%" height="60%">
</p>

Step2B(4):
>At the same time, two new entries are inserted into the ST; one for each side of the branch 2b . The active mask in each entry represents threads that execute the corresponding side of the branch.

We should pop out the SIMT stack structure, which contains the reconvergence item, and move it to the reconvergence table since we reuse the SIMT stack structure as a split table.
```cpp
m_stack.pop_back();
```

<p align="center">
    <img src="/resource/mpipdom/image/st_item.png" width="70%" height="70%">
</p>

Then add two new items to the split table.

```cpp
//divergence path 1
m_stack.push_back(simt_stack_entry());

// divergence path 2
m_stack.push_back(simt_stack_entry());
```
The new items in the split table should record the R-Index which is the index of its corresponding reconvergence item in the reconvergence table.

```cpp
 update the current top of pdom stack
m_stack.back().m_pc = tmp_next_pc;
m_stack.back().m_active_mask = tmp_active_mask;
if (warp_diverged) {
  m_stack.back().m_calldepth = 0;
  m_stack.back().m_recvg_pc = new_recvg_pc;
  //GPGPULearning:ZSY_MPIPDOM:[BEGIN]
  m_stack.back().m_r_index = real_rt_table.size() - 1;
  //GPGPULearning:ZSY_MPIPDOM:[END]
} 
else 
{
  m_stack.back().m_recvg_pc = top_recvg_pc;
}
```

**TODO SPLIT WARP**

Step3A：
>On the clock cycle 3, warp splits B0101 and C1010 are eligible to be scheduled on the pipeline independently. We assume that the scheduler interleaves the available warp splits. Warp splits B0101 and C1010 hide each others’ latency leaving no idle cycles (cycles 3-5). On cycle 6, warp split B0101 reaches the reconvergence point (D) first. 

```cpp
if (tmp_next_pc == top_recvg_pc && (top_type != STACK_ENTRY_TYPE_CALL))
{
    //......
}
```

We have arrived at the reconvergence PC shown in Step2B (1).
<p align="center">
    <img src="/resource/mpipdom/image/reco_pc.png" width="60%" height="60%">
</p>

Step3B：
>Therefore, its entry in the ST table is invalidated, and its active mask is subtracted from the pending active mask of the corresponding entry in the RT table

Retrieve the corresponding R-Index (reconvergence table index) first.

```cpp
int r_index = m_stack.back().m_r_index;
```
<p align="center">
    <img src="/resource/mpipdom/image/3b_rindex.png" width="60%" height="60%">
</p>

```cpp
simt_mask_t st_active_mask = m_stack.back().m_active_mask;
real_rt_table[r_index].m_pending_mask &= ~st_active_mask;
```
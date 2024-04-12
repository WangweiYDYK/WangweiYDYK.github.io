+++
title = 'Compute Sanitizer'
date = 2024-02-26T18:26:30+08:00
draft = true
+++

# Compute Sanitizer

## Memcheck Tool

## Racecheck Tool
The primary use of this tool is to help identify memory access race conditions in CUDA applications that use shared memory.
用于检测是否加入__syncthreads()和__syncwarp()

## Initcheck Tool
 identify when device global memory is accessed without it being initialized via device side writes, or via CUDA memcpy and memset API calls.
## Synccheck Tool
The synccheck tool is a runtime tool that can identify whether a CUDA application is correctly using synchronization primitives, specifically __syncthreads() and __syncwarp() intrinsics and their Cooperative Groups API counterparts.
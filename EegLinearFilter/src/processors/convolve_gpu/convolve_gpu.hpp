//
//  convolve_gpu.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#ifndef CONVOLVE_GPU_HPP
#define CONVOLVE_GPU_HPP

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "../../config.h"
#include "../../data_types.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <chrono>

struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::ComputePipelineState* pipelineState = nullptr;

    MetalContext() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Error: Failed to create Metal device!");
        }
        commandQueue = device->newCommandQueue();
        
        NS::Error* error = nullptr;
        MTL::Library* library = device->newDefaultLibrary();
        if (!library) {
            throw std::runtime_error("Error: Default library not found!");
        }
        
        NS::String* functionName = NS::String::string("convolve_kernel", NS::UTF8StringEncoding);
        MTL::Function* convolutionFunction = library->newFunction(functionName);
        
        if (!convolutionFunction) {
            throw std::runtime_error("Error: Function 'convolve_kernel' not found in .metallib!");
        }
        
        pipelineState = device->newComputePipelineState(convolutionFunction, &error);
        
        if (!pipelineState) {
            throw std::runtime_error("Pipeline creation failed");
        }

        functionName->release();
        convolutionFunction->release();
        library->release();
    }
    
    ~MetalContext() {
        if (pipelineState) pipelineState->release();
        if (commandQueue) commandQueue->release();
        if (device) device->release();
    }
    
    static MetalContext& get() {
        static MetalContext instance;
        return instance;
    }
};

template <int Radius>
ProcessingStats convolve_gpu(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    auto start_wall = std::chrono::high_resolution_clock::now();

    constexpr size_t KSizeConst = 2 * Radius + 1;
    uint32_t KernelSize = (uint32_t)KSizeConst;
    uint32_t outSize = (uint32_t)(data.size() - KSizeConst + 1);
    uint32_t rawDataSize = (uint32_t)data.size();
    
    MetalContext& ctx = MetalContext::get();

    auto mem_start = std::chrono::high_resolution_clock::now();
    
    MTL::Buffer* dataBuffer = ctx.device->newBuffer(
        (void*)data.data(),
        data.size() * sizeof(float),
        MTL::ResourceStorageModeShared,
        nullptr
    );
    
    MTL::Buffer* outBuffer = ctx.device->newBuffer(
        (void*)outputBuffer.data(),
        outputBuffer.size() * sizeof(float),
        MTL::ResourceStorageModeShared,
        nullptr
    );
    
    MTL::Buffer* kernelBuffer = ctx.device->newBuffer(
        convolutionKernel.data(),
        convolutionKernel.size() * sizeof(float),
        MTL::ResourceStorageModeShared
    );
    
    auto mem_end = std::chrono::high_resolution_clock::now();
    double memoryTime = std::chrono::duration<double>(mem_end - mem_start).count();
    
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    computeEncoder->setComputePipelineState(ctx.pipelineState);
    
    computeEncoder->setBuffer(dataBuffer, 0, 0);
    computeEncoder->setBuffer(outBuffer, 0, 1);
    computeEncoder->setBuffer(kernelBuffer, 0, 2);
    computeEncoder->setBytes(&KernelSize, sizeof(uint32_t), 3);
    computeEncoder->setBytes(&outSize, sizeof(uint32_t), 4);
    computeEncoder->setBytes(&rawDataSize, sizeof(uint32_t), 5);
    
    NS::UInteger threadgroupMemSize = (TILE_SIZE + KERNEL_SEGMENT_SIZE) * sizeof(float);
    computeEncoder->setThreadgroupMemoryLength(threadgroupMemSize, 0);

    NS::UInteger numGroups = (outSize + TILE_SIZE - 1) / TILE_SIZE;
    MTL::Size groupSize = MTL::Size::Make(THREADS_PER_GROUP, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(numGroups * THREADS_PER_GROUP, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, groupSize);
    computeEncoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    double gpuStart = commandBuffer->GPUStartTime();
    double gpuEnd = commandBuffer->GPUEndTime();
    double computeTime = gpuEnd - gpuStart;

    if (computeTime < 0) computeTime = 0;

    pool->release();

    mem_start = std::chrono::high_resolution_clock::now();

    dataBuffer->release();
    outBuffer->release();
    kernelBuffer->release();
    
    mem_end = std::chrono::high_resolution_clock::now();
    memoryTime += std::chrono::duration<double>(mem_end - mem_start).count();
    
    auto end_wall = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end_wall - start_wall).count();

    return {
        totalTime,
        computeTime,
        totalTime - computeTime - memoryTime,
        0.0,
        memoryTime
    };
}

#endif // CONVOLVE_GPU_HPP

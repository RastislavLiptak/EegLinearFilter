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
#include <string>

struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    
    MTL::ComputePipelineState* pipelineStateNaive = nullptr;
    MTL::ComputePipelineState* pipelineState32 = nullptr;
    MTL::ComputePipelineState* pipelineState16 = nullptr;

    MetalContext() {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Error: Failed to create Metal device!");
        }
        commandQueue = device->newCommandQueue();
        
        MTL::Library* library = device->newDefaultLibrary();
        if (!library) {
            throw std::runtime_error("Error: Default library not found!");
        }
        
        pipelineStateNaive = createPipeline(library, "convolve_kernel_naive");
        pipelineState32 = createPipeline(library, "convolve_kernel_32");
        pipelineState16 = createPipeline(library, "convolve_kernel_16");
        
        library->release();
    }
    
    MTL::ComputePipelineState* createPipeline(MTL::Library* library, const char* functionNameStr) {
        NS::Error* error = nullptr;
        NS::String* functionName = NS::String::string(functionNameStr, NS::UTF8StringEncoding);
        MTL::Function* convolutionFunction = library->newFunction(functionName);
        
        if (!convolutionFunction) {
            throw std::runtime_error("Error: Function '" + std::string(functionNameStr) + "' not found in .metallib!");
        }
        
        MTL::ComputePipelineState* pso = device->newComputePipelineState(convolutionFunction, &error);
        
        if (!pso) {
            throw std::runtime_error("Pipeline creation failed for " + std::string(functionNameStr));
        }

        functionName->release();
        convolutionFunction->release();
        return pso;
    }
    
    ~MetalContext() {
        if (pipelineStateNaive) pipelineStateNaive->release();
        if (pipelineState32) pipelineState32->release();
        if (pipelineState16) pipelineState16->release();
        if (commandQueue) commandQueue->release();
        if (device) device->release();
    }
    
    static MetalContext& get() {
        static MetalContext instance;
        return instance;
    }
};

template <int Radius>
ProcessingStats convolve_gpu_naive(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    auto start_wall = std::chrono::high_resolution_clock::now();

    constexpr size_t KSizeConst = 2 * Radius + 1;
    uint32_t KernelSize = (uint32_t)KSizeConst;
    uint32_t outSize = (uint32_t)(data.size() - KSizeConst + 1);
    
    MetalContext& ctx = MetalContext::get();
    
    if (!ctx.pipelineStateNaive) {
         throw std::runtime_error("Naive pipeline state is null!");
    }

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
    
    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    
    computeEncoder->setComputePipelineState(ctx.pipelineStateNaive);
    
    computeEncoder->setBuffer(dataBuffer, 0, 0);
    computeEncoder->setBuffer(outBuffer, 0, 1);
    computeEncoder->setBuffer(kernelBuffer, 0, 2);
    computeEncoder->setBytes(&KernelSize, sizeof(uint32_t), 3);
    computeEncoder->setBytes(&outSize, sizeof(uint32_t), 4);
    
    const int items_per_thread = 4;
    NS::UInteger threadsPerGroup = THREADS_PER_GROUP;
    
    NS::UInteger tilePixels = threadsPerGroup * items_per_thread;

    NS::UInteger threadgroupMemSize = (tilePixels + KSizeConst - 1) * sizeof(float);
    computeEncoder->setThreadgroupMemoryLength(threadgroupMemSize, 0);

    NS::UInteger numGroups = (outSize + tilePixels - 1) / tilePixels;
    MTL::Size groupSize = MTL::Size::Make(threadsPerGroup, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(numGroups * threadsPerGroup, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, groupSize);
    computeEncoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    double gpuStart = commandBuffer->GPUStartTime();
    double gpuEnd = commandBuffer->GPUEndTime();
    double computeTime = gpuEnd - gpuStart;
    if (computeTime < 0) computeTime = 0;

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

template <int Radius>
ProcessingStats convolve_gpu(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, bool useHalfPrecision = false) {
    auto start_wall = std::chrono::high_resolution_clock::now();

    constexpr size_t KSizeConst = 2 * Radius + 1;
    uint32_t KernelSize = (uint32_t)KSizeConst;
    uint32_t outSize = (uint32_t)(data.size() - KSizeConst + 1);
    uint32_t rawDataSize = (uint32_t)data.size();
    
    MetalContext& ctx = MetalContext::get();
    
    MTL::ComputePipelineState* currentPipeline = useHalfPrecision ? ctx.pipelineState16 : ctx.pipelineState32;
    
    if (!currentPipeline) {
         throw std::runtime_error("Requested pipeline state is null!");
    }

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
    
    computeEncoder->setComputePipelineState(currentPipeline);
    
    computeEncoder->setBuffer(dataBuffer, 0, 0);
    computeEncoder->setBuffer(outBuffer, 0, 1);
    computeEncoder->setBuffer(kernelBuffer, 0, 2);
    computeEncoder->setBytes(&KernelSize, sizeof(uint32_t), 3);
    computeEncoder->setBytes(&outSize, sizeof(uint32_t), 4);
    computeEncoder->setBytes(&rawDataSize, sizeof(uint32_t), 5);
    
    size_t elementSize = useHalfPrecision ? sizeof(uint16_t) : sizeof(float);
    NS::UInteger threadgroupMemSize = (TILE_SIZE + KERNEL_SEGMENT_SIZE) * elementSize;
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

//
//  convolve_gpu.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#ifndef CONVOLVE_GPU_HPP
#define CONVOLVE_GPU_HPP

#include "../../data_types.hpp"
#include <vector>
#include <iostream>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

struct MetalContext {
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::ComputePipelineState* pipelineState = nullptr;

    MetalContext() {
        device = MTL::CreateSystemDefaultDevice();
        commandQueue = device->newCommandQueue();
        NS::Error* error = nullptr;
        MTL::Library* library = device->newDefaultLibrary();
        
        NS::String* functionName = NS::String::string("convolve_corrected", NS::UTF8StringEncoding);
        MTL::Function* convolutionFunction = library->newFunction(functionName);
        
        if (!convolutionFunction) {
             std::cerr << "Error: Function 'convolve_corrected' not found!" << std::endl;
             exit(1);
        }
        pipelineState = device->newComputePipelineState(convolutionFunction, &error);
        functionName->release(); convolutionFunction->release(); library->release();
    }
    static MetalContext& get() { static MetalContext instance; return instance; }
};

inline void init_gpu_resources() { MetalContext::get(); }

template <int Radius, int ChunkSize>
void convolve_gpu(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    MetalContext& ctx = MetalContext::get();

    MTL::Buffer* dataBuffer = ctx.device->newBuffer((void*)data.data(), data.size() * sizeof(float), MTL::ResourceStorageModeShared, nullptr);
    MTL::Buffer* outBuffer = ctx.device->newBuffer((void*)outputBuffer.data(), outputBuffer.size() * sizeof(float), MTL::ResourceStorageModeShared, nullptr);
    MTL::Buffer* kernelBuffer = ctx.device->newBuffer(convolutionKernel.data(), convolutionKernel.size() * sizeof(float), MTL::ResourceStorageModeShared, nullptr);

    MTL::CommandBuffer* commandBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    computeEncoder->setComputePipelineState(ctx.pipelineState);
    
    constexpr size_t KSizeConst = 2 * Radius + 1;
    uint32_t kernelSize = (uint32_t)KSizeConst;
    uint32_t outSize = (uint32_t)(data.size() - KSizeConst + 1);
    
    computeEncoder->setBuffer(dataBuffer, 0, 0);
    computeEncoder->setBuffer(outBuffer, 0, 1);
    computeEncoder->setBuffer(kernelBuffer, 0, 2);
    computeEncoder->setBytes(&kernelSize, sizeof(uint32_t), 3);
    computeEncoder->setBytes(&outSize, sizeof(uint32_t), 4);

    const int THREADS_PER_GROUP = 256;
    const int ITEMS_PER_THREAD = 4;
    const int TILE_SIZE = THREADS_PER_GROUP * ITEMS_PER_THREAD;
    
    NS::UInteger threadgroupMemSize = (TILE_SIZE + kernelSize - 1) * sizeof(float);
    computeEncoder->setThreadgroupMemoryLength(threadgroupMemSize, 0);

    NS::UInteger numGroups = (outSize + TILE_SIZE - 1) / TILE_SIZE;
    
    MTL::Size groupSize = MTL::Size::Make(THREADS_PER_GROUP, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(numGroups * THREADS_PER_GROUP, 1, 1);
    
    computeEncoder->dispatchThreads(gridSize, groupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    dataBuffer->release(); outBuffer->release(); kernelBuffer->release();
}

#endif // CONVOLVE_GPU_HPP

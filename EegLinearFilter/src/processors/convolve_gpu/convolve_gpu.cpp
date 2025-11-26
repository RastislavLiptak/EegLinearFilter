//
//  convolve_gpu.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "convolve_gpu.hpp"
#include <iostream>


class MetalContext {
public:
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::ComputePipelineState* pipelineState = nullptr;
    bool isValid = false;

    static MetalContext& getInstance() {
        static MetalContext instance;
        return instance;
    }

private:
    MetalContext() {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal device not found" << std::endl;
            pool->release();
            return;
        }

        commandQueue = device->newCommandQueue();

        MTL::Library* library = device->newDefaultLibrary();
        if (!library) {
            std::cerr << "Failed to load default library. Did you compile kernels.metal?" << std::endl;
            pool->release();
            return;
        }

        NS::String* funcName = NS::String::string("convolution_kernel", NS::UTF8StringEncoding);
        MTL::Function* kernelFunction = library->newFunction(funcName);
        
        if (!kernelFunction) {
            std::cerr << "Kernel function not found" << std::endl;
            library->release();
            pool->release();
            return;
        }

        NS::Error* error = nullptr;
        pipelineState = device->newComputePipelineState(kernelFunction, &error);
        
        if (!pipelineState) {
            std::cerr << "Failed to create pipeline state: "
                      << error->localizedDescription()->utf8String() << std::endl;
        } else {
            isValid = true;
        }

        kernelFunction->release();
        library->release();
        pool->release();
    }
    
    ~MetalContext() {
        if (pipelineState) pipelineState->release();
        if (commandQueue) commandQueue->release();
        if (device) device->release();
    }
};

void run_metal_convolution_impl(const float* src, float* dst, size_t dataSize, const float* kernel, size_t kernelSize) {
    MetalContext& ctx = MetalContext::getInstance();
    if (!ctx.isValid) return;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Buffer* inBuffer = ctx.device->newBuffer(src, dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* outBuffer = ctx.device->newBuffer(dst, dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* kernBuffer = ctx.device->newBuffer(kernel, kernelSize * sizeof(float), MTL::ResourceStorageModeShared);

    MTL::CommandBuffer* cmdBuffer = ctx.commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmdBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(ctx.pipelineState);
    encoder->setBuffer(inBuffer, 0, 0);
    encoder->setBuffer(outBuffer, 0, 1);
    encoder->setBuffer(kernBuffer, 0, 2);

    uint32_t kSize = (uint32_t)kernelSize;
    uint32_t dSize = (uint32_t)dataSize;
    encoder->setBytes(&kSize, sizeof(uint32_t), 3);
    encoder->setBytes(&dSize, sizeof(uint32_t), 4);

    NS::UInteger outputCount = dataSize - kernelSize + 1;
    MTL::Size gridSize = MTL::Size(outputCount, 1, 1);
    
    NS::UInteger threadGroupSize = ctx.pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > outputCount) threadGroupSize = outputCount;
    MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();

    cmdBuffer->commit();
    cmdBuffer->waitUntilCompleted();

    memcpy(dst, outBuffer->contents(), dataSize * sizeof(float));

    inBuffer->release();
    outBuffer->release();
    kernBuffer->release();
    
    pool->release();
}

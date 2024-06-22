
#import <quickcluster/device/MetalDeviceManager.h>

@implementation MetalDeviceManager {
    
    id<MTLDevice> _currentDevice;
    id<MTLFunction> _function;
}

+(id<MTLDevice>)defaultDevice {
    return MTLCreateSystemDefaultDevice();
}

- (instancetype)initWithDevice:(id<MTLDevice>)device metallib:(NSString*)path
{
    if (self = [super init]) {
        
        _currentDevice = [MetalDeviceManager defaultDevice];
        
        // Find the metal library
        NSError *error = nil;

        NSURL *mtl_path = [NSURL URLWithString:@"lib/cluster.metallib"];

        if (path != nil) {
            mtl_path = [NSURL URLWithString:path];
        }

        id<MTLLibrary> library = [_currentDevice newLibraryWithURL: mtl_path error: &error];
        
        if (error != nil) {
            NSLog(@"Unable to find metallib file: %@", error);
            return nil;
        }
        
        // Find the function
        _function = [library newFunctionWithName:@"compute_nearest_centroid"];
        assert(_function != nil);
    
    }
    return self;
}

-(int)computeNearestCentroids:(const float *)data centroids:(const float *)centroids context:(const DataContext*)context results:(NSUInteger*)results {

    NSUInteger featureCount = context->rows;
    NSUInteger dims = context->cols;
    NSUInteger k = context->k;

    // Get the pipeline
    NSError *error = nil;
    id<MTLComputePipelineState> pipelineState = [_currentDevice newComputePipelineStateWithFunction:_function error:&error];

    if (pipelineState == nil) {
        NSLog(@"Unable to create new pipeline from function");
        return -1;
    }
    
    // Command queue
    id<MTLCommandQueue> queue = [_currentDevice newCommandQueue];
    
    // Command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    if (commandBuffer == nil) {
        NSLog(@"Unable to get command buffer");
        return -1;
    }
    
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    
    if (commandEncoder == nil) {
        NSLog(@"Unable to get command encoder");
        return -1;
    }
    
    NSUInteger centroidBufferSize = k * dims * sizeof(float);
    NSUInteger dataBufferSize = featureCount * dims * sizeof(float);
    NSUInteger resultBufferSize = featureCount * sizeof(NSUInteger);
    
    id<MTLBuffer> mtlDataBuffer = [_currentDevice newBufferWithLength: dataBufferSize options:MTLResourceStorageModeShared];
    memcpy(mtlDataBuffer.contents, data, dataBufferSize);
    
    id<MTLBuffer> result = [_currentDevice newBufferWithLength:resultBufferSize options:MTLResourceStorageModeShared];
    
    // Encode all the args to pass to the GPU
    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:result offset:0 atIndex:0];
    [commandEncoder setBuffer:mtlDataBuffer offset:0 atIndex:1];
    [commandEncoder setBytes:centroids length:centroidBufferSize atIndex:2];
    [commandEncoder setBytes:context length:sizeof(DataContext) atIndex:3];

    NSUInteger minThreadsInGroup = MIN(pipelineState.maxTotalThreadsPerThreadgroup, featureCount);
    
    // Calculate the number of threads to start
    MTLSize threads = MTLSizeMake(featureCount, 1, 1);
    MTLSize threadGroupSizes = MTLSizeMake(minThreadsInGroup, 1, 1);
    [commandEncoder dispatchThreads:threads threadsPerThreadgroup:threadGroupSizes];
    
    // End encoding
    [commandEncoder endEncoding];
    
    // Call the kernel
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy over the result
    NSUInteger *resultsCasted = (NSUInteger*)result.contents;

    for (size_t i = 0; i < featureCount; i++) {
        results[i] = resultsCasted[i];
    }

    // Finish with success
    return 0;
}

-(void)dealloc {
    // no-op for now
}

@end
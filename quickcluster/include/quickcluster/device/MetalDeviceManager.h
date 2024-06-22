#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Metal/Metal.h>

#import <quickcluster/device/common.h>

NS_ASSUME_NONNULL_BEGIN

@interface MetalDeviceManager: NSObject

+(id<MTLDevice>)defaultDevice;

-(instancetype)initWithDevice:(id<MTLDevice>)device metallib:(NSString*)path;

/// Computes the nearest centroids
-(int)computeNearestCentroids:(const float *)data centroids:(const float *)centroids context:(const DataContext*)context results:(NSUInteger*)results;

@end

NS_ASSUME_NONNULL_END

#include <metal_stdlib>
#include <quickcluster/device/common.h>

using namespace metal;

#define TID [[thread_position_in_grid]]


inline float euclidean_distance(const device float *vec1, const device float *vec2, size_t N) {
    
    float sum = 0.0;
    
    for (size_t i = 0; i < N; i++) {
        sum += pow(vec1[i] - vec2[i], 2);
    }
    
    return sum / static_cast<float>(N);
}


kernel void compute_nearest_centroid(device size_t* result, const device float *data, const device float *centroids, const device DataContext *ctx, uint tid TID) {
    
    size_t centroid_index = 0;
    float min_distance = 0.0;
        
    // For each centroid
    for (size_t index = 0; index < ctx->k; index++) {
        
        size_t offset = tid * ctx->cols;
        size_t centroid_offset = index * ctx->cols;
        
        float dist = euclidean_distance(data + offset, centroids + centroid_offset, ctx->cols);
        
        if (dist < min_distance || min_distance == 0.0) {
            min_distance = dist;
            centroid_index = index; 
        }
    }
    
    result[tid] = centroid_index;
}
#ifndef PIPELINE_MULTI_GPU_H
#define PIPELINE_MULTI_GPU_H

#include "pipeline_one_stage.h"
#include <vector>
#include <thread>
#include <atomic>
#include <map>
#include <string>

// 单卡处理器封装
class SingleGPUProcessor
{
private:
    int _gpu_id;
    CUcontext _cu_context;
    std::vector<RtspUrlParams> _rtsp_sources;
    PipeLineProcessorOneStage* _pipeline;
    std::thread _processing_thread;
    std::atomic<bool> _running{false};

public:
    SingleGPUProcessor(int gpu_id, 
                      const std::vector<RtspUrlParams>& rtsp_sources,
                      const ModelParams& model_params,
                      int jpeg_maxBatch,
                      int jpeg_qsz,
                      int dec_qsz);
    
    ~SingleGPUProcessor();
    
    int StartProcessing();
    int StopProcessing();
    
    // 获取统计信息
    size_t GetDecFrameNumber();
    size_t GetJpegFrameNumber();
    size_t GetYoloV5PreFrameNumber();
    size_t GetYoloV5InferFrameNumber();
    
    // 获取队列状态
    size_t GetDecQWaitNum();
    size_t GetDecQDropNum();
    size_t GetJpegQWaitNum();
    size_t GetJpegQDropNum();
    size_t GetYoloV5PreQWaitNum();
    size_t GetYoloV5PreQDropNum();
    size_t GetYoloV5ResQWaitNum();
    size_t GetYoloV5ResQDropNum();
    
    // 获取视频处理器统计
    std::vector<DecoderProcessorStats> CollectVideoProcessorStats();
    std::map<int, int> CheckDecoderStatusAndRestart();
    int CollectVideoProcessorNumbers();
    
    int GetGPUId() const { return _gpu_id; }
};

// 多卡处理器主类
class MultiGPUProcessor
{
private:
    std::vector<SingleGPUProcessor*>  _gpu_processors;
    std::vector<std::thread> _monitor_threads;
    std::atomic<bool> _running{false};
    
    // 配置文件路径
    std::string _rtsp_config_path;
    std::string _pipeline_config_path;
    
    // 配置参数
    std::vector<RtspUrlParams> _rtsp_sources;
    ModelParams _model_params;
    int _jpeg_maxBatch;
    int _jpeg_qsz;
    int _dec_qsz;
    int _streams_per_gpu = 1; // 每张卡默认处理的流数量, 可由配置文件覆盖

    // 统计信息
    struct GPUStats
    {
        int gpu_id;
        size_t dec_frames = 0;
        size_t jpeg_frames = 0;
        size_t pre_frames = 0;
        size_t infer_frames = 0;
        size_t dec_queue = 0;
        size_t jpeg_queue = 0;
        size_t pre_queue = 0;
        size_t res_queue = 0;
        size_t dec_drops = 0;
        size_t jpeg_drops = 0;
        size_t pre_drops = 0;
        size_t res_drops = 0;
    };
    
    std::vector<GPUStats> _last_stats;
    std::vector<GPUStats> _current_stats;
    
public:
    MultiGPUProcessor(const std::string& rtsp_config_path,
                     const std::string& pipeline_config_path);
    
    ~MultiGPUProcessor();
    
    int Initialize();
    int StartAllGPUs();
    int StopAllGPUs();
    
    // 监控和统计
    void StartMonitoring();
    void StopMonitoring();
    void PrintStatistics();
    
private:
    void LoadConfiguration();
    void MonitorThread(int gpu_id);
};

#endif // PIPELINE_MULTI_GPU_H

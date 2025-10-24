#include "pipeline_multi_gpu.h"
#include "util.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

using json = nlohmann::json;

// SingleGPUProcessor 实现
SingleGPUProcessor::SingleGPUProcessor(int gpu_id,
                                     const std::vector<RtspUrlParams>& rtsp_sources,
                                     const ModelParams& model_params,
                                     int jpeg_maxBatch,
                                     int jpeg_qsz,
                                     int dec_qsz)
    : _gpu_id(gpu_id)
    , _rtsp_sources(rtsp_sources)
{
    // 初始化CUDA上下文
    _cu_context = Init(_gpu_id);
    
    // 创建单卡pipeline
    _pipeline = new PipeLineProcessorOneStage(
        _cu_context, _gpu_id, jpeg_maxBatch, jpeg_qsz, dec_qsz, rtsp_sources, model_params);
}

SingleGPUProcessor::~SingleGPUProcessor()
{
    StopProcessing();
    
    if (_pipeline)
    {
        delete _pipeline;
        _pipeline = nullptr;
    }
    
    if (_cu_context)
    {
        checkCudaErrors(cuCtxDestroy(_cu_context));
    }
}

int SingleGPUProcessor::StartProcessing()
{
    if (_running.load())
    {
        logger->warn("[GPU {}]: Already running", _gpu_id);
        return 0;
    }
    
    _running.store(true);
    
    // 启动pipeline
    _pipeline->StartPipeline();
    
    // 启动处理线程
    _processing_thread = std::thread([this]() {
        logger->info("[GPU {}]: Processing thread started", _gpu_id);
        
        while (_running.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        logger->info("[GPU {}]: Processing thread stopped", _gpu_id);
    });
    
    logger->info("[GPU {}]: Started processing {} video streams", _gpu_id, _rtsp_sources.size());
    return 0;
}

int SingleGPUProcessor::StopProcessing()
{
    if (!_running.load())
    {
        return 0;
    }
    
    _running.store(false);
    
    if (_processing_thread.joinable())
    {
        _processing_thread.join();
    }
    
    logger->info("[GPU {}]: Stopped processing", _gpu_id);
    return 0;
}

// 统计信息获取方法
size_t SingleGPUProcessor::GetDecFrameNumber() 
{ 
    return _pipeline ? _pipeline->GetDecFrameNumber() : 0; 
}

size_t SingleGPUProcessor::GetJpegFrameNumber() 
{ 
    return _pipeline ? _pipeline->GetJpegFrameNumber() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5PreFrameNumber() 
{ 
    return _pipeline ? _pipeline->GetYoloV5PreFrameNumber() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5InferFrameNumber() 
{ 
    return _pipeline ? _pipeline->GetYoloV5InferFrameNumber() : 0; 
}

size_t SingleGPUProcessor::GetDecQWaitNum() 
{ 
    return _pipeline ? _pipeline->GetDecQWaitNum() : 0; 
}

size_t SingleGPUProcessor::GetDecQDropNum() 
{ 
    return _pipeline ? _pipeline->GetDecQDropNum() : 0; 
}

size_t SingleGPUProcessor::GetJpegQWaitNum() 
{ 
    return _pipeline ? _pipeline->GetJpegQWaitNum() : 0; 
}

size_t SingleGPUProcessor::GetJpegQDropNum() 
{ 
    return _pipeline ? _pipeline->GetJpegQDropNum() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5PreQWaitNum() 
{ 
    return _pipeline ? _pipeline->GetYoloV5PreQWaitNum() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5PreQDropNum() 
{ 
    return _pipeline ? _pipeline->GetYoloV5PreQDropNum() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5ResQWaitNum() 
{ 
    return _pipeline ? _pipeline->GetYoloV5ResQWaitNum() : 0; 
}

size_t SingleGPUProcessor::GetYoloV5ResQDropNum() 
{ 
    return _pipeline ? _pipeline->GetYoloV5ResQDropNum() : 0; 
}

std::vector<DecoderProcessorStats> SingleGPUProcessor::CollectVideoProcessorStats()
{
    return _pipeline ? _pipeline->CollectVideoProcessorStats() : std::vector<DecoderProcessorStats>();
}

std::map<int, int> SingleGPUProcessor::ChcekDecoderStautsAndRestart()
{
    return _pipeline ? _pipeline->ChcekDecoderStautsAndRestart() : std::map<int, int>();
}

int SingleGPUProcessor::CollectVideoProcessorNumbers()
{
    return _pipeline ? _pipeline->CollectVideoProcessorNumbers() : 0;
}

// MultiGPUProcessor 实现
MultiGPUProcessor::MultiGPUProcessor(const std::string& rtsp_config_path,
                                   const std::string& pipeline_config_path)
    : _rtsp_config_path(rtsp_config_path)
    , _pipeline_config_path(pipeline_config_path)
{
}

MultiGPUProcessor::~MultiGPUProcessor()
{
    StopAllGPUs();
    
    for (auto* processor : _gpu_processors)
    {
        delete processor;
    }
    _gpu_processors.clear();
}

int MultiGPUProcessor::Initialize()
{
    // 加载配置
    LoadConfiguration();
    
    // 获取GPU数量
    int gpu_count = 0;
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&gpu_count));
    
    logger->info("Detected {} GPUs", gpu_count);
    
    // 为每个GPU创建处理器, 支持每张卡多路流
    for (int i = 0; i < gpu_count; i++)
    {
        // 每个GPU分配一路视频流
        std::vector<RtspUrlParams> single_gpu_sources;
        int start_index = i * _streams_per_gpu;
        int end_index = std::min(start_index + _streams_per_gpu, (int)_rtsp_sources.size());
        if (start_index >= (int)_rtsp_sources.size())
        {
            logger->warn("GPU {} has no video source assigned", i);
            continue;
        }
        for (int idx = start_index; idx < end_index; ++idx)
        {
            single_gpu_sources.push_back(_rtsp_sources[idx]);
        }
        SingleGPUProcessor* processor = new SingleGPUProcessor(
            i, single_gpu_sources, _model_params, 
            _jpeg_maxBatch, _jpeg_qsz, _dec_qsz);
        
        _gpu_processors.push_back(processor);
        
        // 初始化统计信息
        GPUStats stats; 
        stats.gpu_id = i; 
        _last_stats.push_back(stats); 
        _current_stats.push_back(stats);
        logger->info("GPU {} assigned {} streams (source index range: [{} - {}))", i, single_gpu_sources.size(), start_index, end_index);

    }
    
    logger->info("Initialized {} GPU processors", _gpu_processors.size());
    return 0;
}

int MultiGPUProcessor::StartAllGPUs()
{
    _running.store(true);
    
    // 启动所有GPU处理器
    for (auto* processor : _gpu_processors)
    {
        processor->StartProcessing();
    }
    
    // 启动监控线程
    StartMonitoring();
    
    logger->info("Started all {} GPU processors", _gpu_processors.size());
    return 0;
}

int MultiGPUProcessor::StopAllGPUs()
{
    _running.store(false);
    
    // 停止监控线程
    StopMonitoring();
    
    // 停止所有GPU处理器
    for (auto* processor : _gpu_processors)
    {
        processor->StopProcessing();
    }
    
    logger->info("Stopped all GPU processors");
    return 0;
}

void MultiGPUProcessor::LoadConfiguration()
{
    // 加载RTSP配置
    RtspUrlManager rtsp_manager(_rtsp_config_path);
    _rtsp_sources = rtsp_manager.getUrls();
    
    // 加载pipeline配置
    std::ifstream pipe_file(_pipeline_config_path);
    json pipe_json;
    pipe_file >> pipe_json;
    
    _jpeg_maxBatch = pipe_json["jpeg_maxBatch"].get<int>();
    _jpeg_qsz = pipe_json["jpeg_qsz"].get<int>();
    _dec_qsz = pipe_json["dec_qsz"].get<int>();
    if (pipe_json.contains("streams_per_gpu")) {
        _streams_per_gpu = pipe_json["streams_per_gpu"].get<int>();
    }
    
    // 加载模型参数
    _model_params = pipe_json["yolov5s_params"].get<ModelParams>();
    
    // 生成engine文件路径
    if (_model_params.engine_file.empty())
    {
        _model_params.engine_file = generateEnginePath(_model_params.onnx_file);
    }
    
    logger->info("Loaded configuration: {} video sources, streams_per_gpu={} ", _rtsp_sources.size(), _streams_per_gpu);
}

void MultiGPUProcessor::StartMonitoring()
{
    // 为每个GPU创建监控线程
    for (size_t i = 0; i < _gpu_processors.size(); i++)
    {
        _monitor_threads.emplace_back(&MultiGPUProcessor::MonitorThread, this, i);
    }
}

void MultiGPUProcessor::StopMonitoring()
{
    // 等待所有监控线程结束
    for (auto& thread : _monitor_threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    _monitor_threads.clear();
}

void MultiGPUProcessor::MonitorThread(int gpu_id)
{
    logger->info("[GPU {}]: Monitor thread started", gpu_id);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t time_interval = 10; // 10秒间隔
    
    while (_running.load())
    {
        std::this_thread::sleep_for(std::chrono::seconds(time_interval));
        
        if (gpu_id < _gpu_processors.size())
        {
            SingleGPUProcessor* processor = _gpu_processors[gpu_id];
            
            // 收集当前统计信息
            GPUStats& current = _current_stats[gpu_id];
            current.gpu_id = gpu_id;
            current.dec_frames = processor->GetDecFrameNumber();
            current.jpeg_frames = processor->GetJpegFrameNumber();
            current.pre_frames = processor->GetYoloV5PreFrameNumber();
            current.infer_frames = processor->GetYoloV5InferFrameNumber();
            current.dec_queue = processor->GetDecQWaitNum();
            current.jpeg_queue = processor->GetJpegQWaitNum();
            current.pre_queue = processor->GetYoloV5PreQWaitNum();
            current.res_queue = processor->GetYoloV5ResQWaitNum();
            current.dec_drops = processor->GetDecQDropNum();
            current.jpeg_drops = processor->GetJpegQDropNum();
            current.pre_drops = processor->GetYoloV5PreQDropNum();
            current.res_drops = processor->GetYoloV5ResQDropNum();
            
            // 打印统计信息
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double total_time = duration.count() / 1000000.0;
            
            // 计算10秒内的增量
            size_t dec_10s = current.dec_frames - _last_stats[gpu_id].dec_frames;
            size_t jpeg_10s = current.jpeg_frames - _last_stats[gpu_id].jpeg_frames;
            size_t pre_10s = current.pre_frames - _last_stats[gpu_id].pre_frames;
            size_t infer_10s = current.infer_frames - _last_stats[gpu_id].infer_frames;
            
            double fps_dec = (double)dec_10s / 10.0;
            double fps_jpeg = (double)jpeg_10s / 10.0;
            double fps_pre = (double)pre_10s / 10.0;
            double fps_infer = (double)infer_10s / 10.0;

            double fps_dec_t = (double)current.dec_frames / total_time;
            double fps_jpeg_t = (double)current.jpeg_frames / total_time;
            double fps_pre_t = (double)current.pre_frames / total_time;
            double fps_infer_t = (double)current.infer_frames / total_time;
            
            std::ostringstream oss;
            oss << "\n"
                << "========================================== GPU " << gpu_id << " ==========================================\n"
                << "     |                 |        frames   |      FPS       |\n"
                << "     |     dec         | " << std::setw(12) << current.dec_frames << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec_t << "   |\n"
                << "     |     jpeg        | " << std::setw(12) << current.jpeg_frames << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg_t << "   |\n"
                << "total|  yolov5s-pre    | " << std::setw(12) << current.pre_frames << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pre_t << "   |\n"
                << "     |  yolov5s-infer  | " << std::setw(12) << current.infer_frames << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_infer_t << "   |\n"
                << "     |     dec         | " << std::setw(12) << dec_10s << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_dec << "   |\n"
                << "     |     jpeg        | " << std::setw(12) << jpeg_10s << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_jpeg << "   |\n"
                << "10s  |  yolov5s-pre    | " << std::setw(12) << pre_10s << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_pre << "   |\n"
                << "     |  yolov5s-infer  | " << std::setw(12) << infer_10s << "    | " << std::setw(12) << std::fixed << std::setprecision(2) << fps_infer << "   |\n"
                << "     |                 |        Now      |      Drops     |\n"
                << "     |     dec         | " << std::setw(12) << current.dec_queue << "    | " << std::setw(12) << current.dec_drops << "   |\n"
                << "queue|     jpeg        | " << std::setw(12) << current.jpeg_queue << "    | " << std::setw(12) << current.jpeg_drops << "   |\n"
                << "     |  yolov5s-pre    | " << std::setw(12) << current.pre_queue << "    | " << std::setw(12) << current.pre_drops << "   |\n"
                << "     |  yolov5s-res    | " << std::setw(12) << current.res_queue << "    | " << std::setw(12) << current.res_drops << "   |\n"
                << "================================================================================\n";
            
            logger->info(oss.str());
            
            // 更新上次统计信息
            _last_stats[gpu_id] = current;
        }
    }
    
    logger->info("[GPU {}]: Monitor thread stopped", gpu_id);
}

void MultiGPUProcessor::PrintStatistics()
{
    std::ostringstream oss;
    oss << "\n==================== Multi-GPU Statistics ====================\n";
    
    for (size_t i = 0; i < _gpu_processors.size(); i++)
    {
        const GPUStats& stats = _current_stats[i];
        oss << "GPU " << i << ": Dec=" << stats.dec_frames 
            << ", JPEG=" << stats.jpeg_frames 
            << ", Pre=" << stats.pre_frames 
            << ", Infer=" << stats.infer_frames << "\n";
    }
    
    oss << "===============================================================\n";
    logger->info(oss.str());
}

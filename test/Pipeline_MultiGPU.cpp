#include "pipeline_multi_gpu.h"

using json = nlohmann::json;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <rtsp config file> <pipeline config file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../config/rtsp_params.json ../config/pipeline_one_stage.json" << std::endl;
        return 1;
    }
    
    initializeLogger();
    
    std::string rtsp_config = argv[1];
    std::string pipeline_config = argv[2];
    
    logger->info("Starting Multi-GPU Pipeline");
    logger->info("RTSP Config: {}", rtsp_config);
    logger->info("Pipeline Config: {}", pipeline_config);
    
    // 创建多GPU处理器
    MultiGPUProcessor* multi_processor = new MultiGPUProcessor(rtsp_config, pipeline_config);
    
    // 初始化
    int ret = multi_processor->Initialize();
    if (ret != 0)
    {
        logger->error("Failed to initialize multi-GPU processor");
        delete multi_processor;
        return 1;
    }
    
    // 启动所有GPU
    ret = multi_processor->StartAllGPUs();
    if (ret != 0)
    {
        logger->error("Failed to start multi-GPU processor");
        delete multi_processor;
        return 1;
    }
    
    logger->info("Multi-GPU pipeline started successfully");
    logger->info("Press Ctrl+C to stop...");
    
    // 主循环
    try
    {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::seconds(60));
            
            // 定期打印总体统计信息
            multi_processor->PrintStatistics();
        }
    }
    catch (const std::exception& e)
    {
        logger->error("Exception in main loop: {}", e.what());
    }
    
    // 清理资源
    logger->info("Stopping multi-GPU pipeline...");
    multi_processor->StopAllGPUs();
    delete multi_processor;
    
    logger->info("Multi-GPU pipeline stopped");
    return 0;
}

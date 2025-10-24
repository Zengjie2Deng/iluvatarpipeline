#include "base_processor.h"
#include "json.h"

struct PPYoloEPreSurfaces
{
    SurfaceCudaBuff                   preSurface;
    std::vector<ViDecSurfaceCudaBuff> oriSurfaces;
};

struct PPYoloEResult
{
    ViDecSurfaceCudaBuff  oriSurfaces;
    std::vector<Rect>     rects;
    std::vector<RectInfo> rectsInfo;
};

class PPYoloEModelProcessor : public ModelProcessor
{
private:
    ProcessQueue<ViDecSurfaceCudaBuff>* _video_queue;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue;
    ProcessQueue<PPYoloEResult>*        _res_queue;
    DetectPreprocessor*                 _m_preprocessor;
    ProcessQueue<PPYoloEPreSurfaces>*   _m_preprocess_queue;

    virtual int InitPreProcess() override;
    virtual int InitIxrtIfner() override;

    virtual void IxrtInfer() override;
    virtual void CudaPreProcess() override;

public:
    PPYoloEModelProcessor(CUcontext                           cu_context,
                          int                                 resize_h,
                          int                                 resize_w,
                          int                                 pre_maxBatch,
                          int                                 pre_qsz,
                          int                                 infer_maxBatch,
                          const std::vector<std::string>&     custom_outputs,
                          const std::string&                  onnx_file,
                          const std::string&                  engine_file,
                          ProcessQueue<ViDecSurfaceCudaBuff>* video_queue,
                          ProcessQueue<ViDecSurfaceCudaBuff>* jpeg_queue,
                          ProcessQueue<PPYoloEResult>*        res_queue);

    ~PPYoloEModelProcessor() override;

    virtual size_t GetPreQWaitNum() override { return _m_preprocess_queue->size(); }
    virtual size_t GetPreQDropNum() override { return _m_preprocess_queue->drops_count(); }
};

class PPLCNetModelProcessor : public ModelProcessor
{
private:
    ProcessQueue<PPYoloEResult>*   _input_queue;
    ClassifyPreprocessor*          _m_preprocessor;
    ProcessQueue<SurfaceCudaBuff>* _m_preprocess_queue;

    virtual int InitPreProcess() override;
    virtual int InitIxrtIfner() override;

    virtual void IxrtInfer() override;
    virtual void CudaPreProcess() override;

public:
    PPLCNetModelProcessor(CUcontext                       cu_context,
                          int                             resize_h,
                          int                             resize_w,
                          int                             pre_maxBatch,
                          int                             pre_qsz,
                          int                             infer_maxBatch,
                          const std::vector<std::string>& custom_outputs,
                          const std::string&              onnx_file,
                          const std::string&              engine_file,
                          ProcessQueue<PPYoloEResult>*    input_queue);

    ~PPLCNetModelProcessor() override;

    virtual size_t GetPreQWaitNum() override { return _m_preprocess_queue->size(); }
    virtual size_t GetPreQDropNum() override { return _m_preprocess_queue->drops_count(); }
};

class PipeLineProcessorTwoStage
{
private:
    int       _dev_id;
    CUcontext _cu_context;

    /* video */
    int                                  _dec_qsz = 0;
    std::vector<RtspUrlParams>           _rtsp_sources;
    std::map<int, VideoStreamProcessor*> _video_processores;
    std::map<int, int>                   _video_reset_count;
    ProcessQueue<ViDecSurfaceCudaBuff>*  _video_queue = nullptr;

    int InitVideoStreams(const int dev_id);

    /* jpeg */
    std::atomic<size_t>                 _p_jpegEncodeFrames = 0;
    int                                 _p_jpeg_maxBatch    = 0;
    int                                 _jpeg_qsz           = 0;
    IluvatarJpegCodec*                  _jpeg_encoder       = nullptr;
    ProcessQueue<ViDecSurfaceCudaBuff>* _jpeg_queue         = nullptr;
    TaskThread                          _p_thread_jpeg;

    int JpegEncode();

    /* ppyoloe */
    PPYoloEModelProcessor*       _ppyoloe_processor = nullptr;
    ProcessQueue<PPYoloEResult>* _ppyoloe_res_queue = nullptr;

    /* pplcnet */
    PPLCNetModelProcessor* _pplcnet_processor = nullptr;

public:
    PipeLineProcessorTwoStage(CUcontext                         cu_context,
                      const int                         dev_id,
                      const int                         jpeg_maxBatch,
                      const int                         jpeg_qsz,
                      const int                         dec_qsz,
                      const std::vector<RtspUrlParams>& rtsp_sources,
                      const ModelParams&                ppyoloe_params,
                      const ModelParams&                pplcnet_params);

    size_t GetDecFrameNumber() { return _video_queue->count(); }
    size_t GetDecQWaitNum() { return _video_queue->size(); }
    size_t GetDecQDropNum() { return _video_queue->drops_count(); }

    size_t GetJpegFrameNumber() { return _p_jpegEncodeFrames.load(); }
    size_t GetJpegQWaitNum() { return _jpeg_queue->size(); }
    size_t GetJpegQDropNum() { return _jpeg_queue->drops_count(); }

    size_t GetPPYoloEPreFrameNumber() { return _ppyoloe_processor->GetPreFrameNumber(); }
    size_t GetPPYoloEInferFrameNumber() { return _ppyoloe_processor->GetInferFrameNumber(); }
    size_t GetPPYoloEPreQWaitNum() { return _ppyoloe_processor->GetPreQWaitNum(); }
    size_t GetPPYoloEPreQDropNum() { return _ppyoloe_processor->GetPreQDropNum(); }

    size_t GetPPYoloEResQWaitNum() { return _ppyoloe_res_queue->size(); }
    size_t GetPPYoloEResQDropNum() { return _ppyoloe_res_queue->drops_count(); }

    size_t GetPPLCNetPreFrameNumber() { return _pplcnet_processor->GetPreFrameNumber(); }
    size_t GetPPLCNetInferFrameNumber() { return _pplcnet_processor->GetInferFrameNumber(); }
    size_t GetPPLCNetPreQWaitNum() { return _pplcnet_processor->GetPreQWaitNum(); }
    size_t GetPPLCNetPreQDropNum() { return _pplcnet_processor->GetPreQDropNum(); }

    std::map<int, int>                 CheckDecoderStatusAndRestart();
    std::vector<DecoderProcessorStats> CollectVideoProcessorStats();
    int                                CollectVideoProcessorNumbers() { return _video_processores.size(); }

    int StartPipeline();
    ~PipeLineProcessorTwoStage();
};
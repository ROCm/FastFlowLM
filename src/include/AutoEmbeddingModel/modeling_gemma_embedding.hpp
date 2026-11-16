/// \file modeling_gemma_embedding.hpp
/// \brief Gemma_Embedding class
/// \author FastFlowLM Team
/// \date 2025-10-23
/// \version 0.9.24
/// \note This is a header file for the Gemma_Embedding class

#include "auto_embedding_model.hpp"

class Gemma_Embedding : public AutoEmbeddingModel{
private:
    std::string _get_task_prefix(const std::string& task) {
        if (task.empty() || task == "query")
            return "task: search result | query: ";
        if (task == "document") return "title: none | text: ";
        if (task == "clustering") return "task: clustering | query: ";
        if (task == "classification")
            return "task: classification | query: ";
        if (task == "code_retrieval") return "task: code retrieval | query: ";
        if (task == "sentence_similarity") return "task: sentence similarity | query: ";
        if (task == "summarization") return "task: summarization | query: ";
        throw std::invalid_argument("Unsupported Gemma embedding task: " + task);
    }
public:
    Gemma_Embedding(flm_rt::device* npu_device_inst);
    ~Gemma_Embedding();

    void load_model(std::string model_path, json model_info, bool enable_preemption = false) override;
    EmbeddingResult embed(const json& request,
                          std::function<bool()> cancelled = [] { return false; }) override;
};
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <unordered_map>

// 打印用法说明
static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -p \"Hello my name is\" -n 32 -t \"H,e,l,o\" -g 10\n", argv[0]);
    LOG_TEE("\n");
}

// 解析传入的字符列表，并将其转换为token
std::vector<llama_token> parse_target_tokens(llama_context * ctx, const std::string & target_chars) {
    std::vector<llama_token> target_tokens;
    std::istringstream ss(target_chars);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::vector<llama_token> tokens = ::llama_tokenize(ctx, token, false);
        target_tokens.insert(target_tokens.end(), tokens.begin(), tokens.end());
    }
    return target_tokens;
}

int main(int argc, char ** argv) {
    gpt_params params;
    std::string target_chars;
    int n_gpu_layers = 99; // 设置默认值为99

    // 设置初始提示语和预测长度
    params.prompt = "Hello my name is";
    params.n_predict = 1;

    // 解析命令行参数，如果解析失败则打印用法并退出
    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    // 解析 -t 参数
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t" && i + 1 < argc) {
            target_chars = argv[i + 1];
        }
    }

    if (target_chars.empty()) {
        LOG_TEE("Error: -t parameter is missing or empty.\n");
        return 1;
    }

    // 初始化LLM
    llama_backend_init();
    llama_numa_init(params.numa);

    // 初始化模型参数
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    // model_params.n_gpu_layers = n_gpu_layers; // 设置存储在VRAM中的层数
    model_params.main_gpu = 0; // 设置使用的主要GPU

    // 加载模型并配置到GPU RAM中
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // 初始化上下文参数并创建上下文
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // 将提示语进行token化
    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    // 确保KV缓存足够大以容纳所有提示语和生成的tokens
    if (tokens_list.size() > n_ctx) {
        LOG_TEE("%s: error: required KV cache size is not big enough\n", __func__);
        return 1;
    }

    // 创建一个大小为512的llama_batch，用于提交token数据进行解码
    llama_batch batch = llama_batch_init(512, 0, 1);

    // 评估初始提示语
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode只会输出提示语最后一个token的logits
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // 获取当前token的logits
    auto * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    auto n_vocab = llama_n_vocab(model);

    // 计算softmax概率
    std::vector<float> probs(n_vocab);
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = exp(logits[i]);
        sum_exp += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] /= sum_exp;
    }

    // 解析目标字符并转换为token
    std::vector<llama_token> target_tokens = parse_target_tokens(ctx, target_chars);

    // 打印目标字符的概率
    for (const auto &token : target_tokens) {
        printf("Token %d: prob = %f, token = %s\n", token, probs[token], llama_token_to_piece(ctx, token).c_str());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}

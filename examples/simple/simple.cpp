#include "common.h"
#include "llama.h"
#include "httplib.h"
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <unordered_map>

using json = nlohmann::json;

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

// 计算softmax概率
std::vector<float> compute_softmax(const float* logits, int n_vocab) {
    std::vector<float> probs(n_vocab);
    float sum_exp = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = exp(logits[i]);
        sum_exp += probs[i];
    }
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] /= sum_exp;
    }
    return probs;
}

void handle_props(const httplib::Request &req, httplib::Response &res, llama_model* model, llama_context* ctx) {
    json json_req;

    try {
        json_req = json::parse(req.body);
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid JSON format\"}", "application/json");
        return;
    }

    // 获取参数
    std::string prompt = json_req["prompt"].get<std::string>();
    std::string target_chars = json_req["target_chars"].get<std::string>();

    if (prompt.empty() || target_chars.empty()) {
        res.status = 400;
        res.set_content("{\"error\": \"Missing prompt or target_chars parameter\"}", "application/json");
        return;
    }

    // 将提示语进行token化
    std::vector<llama_token> tokens_list = ::llama_tokenize(ctx, prompt, true);
    const int n_ctx = llama_n_ctx(ctx);
    if (tokens_list.size() > n_ctx) {
        res.status = 400;
        res.set_content("{\"error\": \"Prompt is too long\"}", "application/json");
        return;
    }

    // 评估初始提示语
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (llama_decode(ctx, batch) != 0) {
        res.status = 500;
        res.set_content("{\"error\": \"Failed to decode\"}", "application/json");
        return;
    }

    // 获取logits和计算softmax概率
    auto * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    auto n_vocab = llama_n_vocab(model);
    std::vector<float> probs = compute_softmax(logits, n_vocab);

    // 解析目标字符并转换为token
    std::vector<llama_token> target_tokens = parse_target_tokens(ctx, target_chars);

    // 构建JSON响应
    json json_res;
    for (const auto &token : target_tokens) {
        json token_info;
        token_info["token"] = llama_token_to_piece(ctx, token);
        token_info["probability"] = probs[token];
        json_res["tokens"].push_back(token_info);
    }

    // 设置响应内容
    res.set_content(json_res.dump(), "application/json");

    // 释放资源
    llama_batch_free(batch);
}

void handle_shutdown(const httplib::Request &req, httplib::Response &res, httplib::Server* svr, llama_context* ctx, llama_model* model) {
    res.set_content("{\"message\": \"Shutting down\"}", "application/json");
    svr->stop();

    // 释放资源
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

int main(int argc, char **argv) {
    httplib::Server svr;

    // 初始化LLM
    llama_backend_init();

    // 初始化参数
    gpt_params params;
    params.model = "model.gguf"; // 确保设置正确的模型路径

    // 解析命令行参数，如果解析失败则打印用法并退出
    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    llama_numa_init(params.numa);

    // 初始化模型参数
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    model_params.main_gpu = 0;

    // 加载模型
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // 初始化上下文
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // 注册POST处理函数
    svr.Post("/props", [&model, &ctx](const httplib::Request &req, httplib::Response &res) {
        handle_props(req, res, model, ctx);
    });

    svr.Post("/shutdown", [&svr, &ctx, &model](const httplib::Request &req, httplib::Response &res) {
        handle_shutdown(req, res, &svr, ctx, model);
    });

    // 设置端口并启动服务器
    svr.listen("0.0.0.0", 8080);

    return 0;
}

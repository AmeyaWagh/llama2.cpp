#ifndef LLAMA2CPP_UTILS_HPP
#define LLAMA2CPP_UTILS_HPP
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <transformers-lite/layers/transformer.hpp>

namespace llama2cpp {
using namespace transformers_lite;

void loadModel(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights<CPU, float32_t> &weights) {
    std::ifstream file(checkpoint_path, std::ios::binary);
    if (!file) {
        std::cerr << "Couldn't open file " << checkpoint_path << '\n';
        std::exit(EXIT_FAILURE);
    }
    file.read(reinterpret_cast<char *>(&config), sizeof(TransformerConfig));
    auto shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = std::abs(config.vocab_size);

    size_t dim = static_cast<size_t>(config.dim);
    size_t head_size = static_cast<size_t>(config.dim / config.n_heads);
    size_t vocab_size = static_cast<size_t>(config.vocab_size);
    size_t n_heads = static_cast<size_t>(config.n_heads);
    size_t n_kv_heads = static_cast<size_t>(config.n_kv_heads);
    size_t hidden_dim = static_cast<size_t>(config.hidden_dim);

    // make sure the multiplications below are done in 64bit to fit the
    // parameter counts of 13B+ models
    unsigned long long n_layers = config.n_layers;

    weights.token_embedding_table.reShape(Shape(vocab_size, dim));
    file.read(reinterpret_cast<char *>(weights.token_embedding_table.data()), weights.token_embedding_table.numBytes());

    weights.rms_att_weight.reShape(Shape(n_layers, dim));
    file.read(reinterpret_cast<char *>(weights.rms_att_weight.data()), weights.rms_att_weight.numBytes());

    weights.wq.reShape(Shape(n_layers, dim, n_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wq.data()), weights.wq.numBytes());

    weights.wk.reShape(Shape(n_layers, dim, n_kv_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wk.data()), weights.wk.numBytes());

    weights.wv.reShape(Shape(n_layers, dim, n_kv_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wv.data()), weights.wv.numBytes());

    weights.wo.reShape(Shape(n_layers, dim, n_heads * head_size));
    file.read(reinterpret_cast<char *>(weights.wo.data()), weights.wo.numBytes());

    weights.rms_ffn_weight.reShape(Shape(n_layers, dim));
    file.read(reinterpret_cast<char *>(weights.rms_ffn_weight.data()), weights.rms_ffn_weight.numBytes());

    weights.w1.reShape(Shape(n_layers, hidden_dim, dim));
    file.read(reinterpret_cast<char *>(weights.w1.data()), weights.w1.numBytes());

    weights.w2.reShape(Shape(n_layers, dim, hidden_dim));
    file.read(reinterpret_cast<char *>(weights.w2.data()), weights.w2.numBytes());

    weights.w3.reShape(Shape(n_layers, hidden_dim, dim));
    file.read(reinterpret_cast<char *>(weights.w3.data()), weights.w3.numBytes());

    weights.rms_final_weight.reShape(Shape(dim));
    file.read(reinterpret_cast<char *>(weights.rms_final_weight.data()), weights.rms_final_weight.numBytes());

    // ptr += config.dim;
    // ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    // ptr += config.seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)
    weights.wcls.reShape(Shape(vocab_size, dim));
    if (!shared_weights) {
        file.seekg((config.dim + config.seq_len * head_size) * sizeof(float32_t), std::ios::cur);
        file.read(reinterpret_cast<char *>(weights.wcls.data()), weights.wcls.numBytes());
    } else {
        weights.wcls.copyFrom(weights.token_embedding_table.data(), weights.wcls.numElements());
    }

    file.close();
}

#ifdef TRANSFORMERS_CUDA_ENABLED
void loadModel(const std::string &checkpoint_path, TransformerConfig &config, TransformerWeights<CUDA, float32_t> &weights){
    //@TODO implement this
}
#endif

// ---------------------------------------------------------------------------
// Safetensors export
// ---------------------------------------------------------------------------

// Permutes Q or K weight rows to match HuggingFace's interleaved RoPE layout.
//
// llama2.c stores each head's rows as [first_half | second_half] (rotation
// applied to the two halves separately). HuggingFace's LlamaRotaryEmbedding
// uses adjacent pairs, so consecutive pairs must be interleaved:
//   dst[h*hs + 2*p]   = src[h*hs + p]           (first-half row p)
//   dst[h*hs + 2*p+1] = src[h*hs + hs/2 + p]    (second-half row p)
//
// This mirrors export.py::permute_original().
static std::vector<float32_t> permuteQK(const float32_t *src,
                                         size_t n_heads,
                                         size_t head_size,
                                         size_t dim_in) {
    const size_t half = head_size / 2;
    std::vector<float32_t> dst(n_heads * head_size * dim_in);
    for (size_t h = 0; h < n_heads; ++h) {
        for (size_t p = 0; p < half; ++p) {
            std::copy_n(src + (h * head_size + p)        * dim_in, dim_in,
                        dst.data() + (h * head_size + 2*p)   * dim_in);
            std::copy_n(src + (h * head_size + half + p) * dim_in, dim_in,
                        dst.data() + (h * head_size + 2*p+1) * dim_in);
        }
    }
    return dst;
}

// Writes model weights to a single .safetensors file with HuggingFace
// LlamaForCausalLM-compatible key names and shapes.
//
// Safetensors format:
//   [8 bytes: little-endian uint64 header size]
//   [N bytes: UTF-8 JSON {"name":{"dtype","shape","data_offsets"},...}]
//   [tensor data: concatenated raw float32 bytes]
//
// Key names match the official Meta Llama 2 HuggingFace release.
// q_proj and k_proj weights are permuted to match HF's interleaved RoPE
// convention (see export.py::permute_original for the Python equivalent).
void saveSafetensors(const std::string &output_path,
                     const TransformerConfig &config,
                     const TransformerWeights<CPU, float32_t> &weights) {
    const size_t dim        = static_cast<size_t>(config.dim);
    const size_t hidden_dim = static_cast<size_t>(config.hidden_dim);
    const size_t n_layers   = static_cast<size_t>(config.n_layers);
    const size_t n_heads    = static_cast<size_t>(config.n_heads);
    const size_t n_kv_heads = static_cast<size_t>(config.n_kv_heads);
    const size_t vocab_size = static_cast<size_t>(std::abs(config.vocab_size));
    const size_t head_size  = dim / n_heads;

    // owned_buf holds permuted copies of q/k weights; data ptr aliases into it.
    struct Entry {
        std::string name;
        const float32_t *data;
        size_t num_bytes;
        std::vector<size_t> shape;
        std::vector<float32_t> owned_buf;
    };

    std::vector<Entry> tensors;
    tensors.reserve(3 + n_layers * 9);

    tensors.push_back({"model.embed_tokens.weight",
                        weights.token_embedding_table.data(),
                        weights.token_embedding_table.numBytes(),
                        {vocab_size, dim}, {}});

    for (size_t i = 0; i < n_layers; ++i) {
        const std::string pfx = "model.layers." + std::to_string(i) + ".";

        auto rms_att = weights.rms_att_weight.slice(i);
        tensors.push_back({pfx + "input_layernorm.weight",
                            rms_att.data(), rms_att.numBytes(), {dim}, {}});

        // q_proj / k_proj: permute rows for HF interleaved RoPE convention.
        // matmulCPU treats W as (out_dim, in_dim):
        //   wq: (n_heads * head_size, dim)
        //   wk: (n_kv_heads * head_size, dim)
        {
            auto wq_i = weights.wq.slice(i);
            auto buf  = permuteQK(wq_i.data(), n_heads, head_size, dim);
            size_t nb = buf.size() * sizeof(float32_t);
            tensors.push_back({pfx + "self_attn.q_proj.weight",
                                nullptr, nb, {n_heads * head_size, dim}, std::move(buf)});
            tensors.back().data = tensors.back().owned_buf.data();
        }
        {
            auto wk_i   = weights.wk.slice(i);
            auto buf    = permuteQK(wk_i.data(), n_kv_heads, head_size, dim);
            size_t nb   = buf.size() * sizeof(float32_t);
            tensors.push_back({pfx + "self_attn.k_proj.weight",
                                nullptr, nb, {n_kv_heads * head_size, dim}, std::move(buf)});
            tensors.back().data = tensors.back().owned_buf.data();
        }

        auto wv_i = weights.wv.slice(i);
        tensors.push_back({pfx + "self_attn.v_proj.weight",
                            wv_i.data(), wv_i.numBytes(), {n_kv_heads * head_size, dim}, {}});

        // wo: out_dim = dim, in_dim = n_heads * head_size
        auto wo_i = weights.wo.slice(i);
        tensors.push_back({pfx + "self_attn.o_proj.weight",
                            wo_i.data(), wo_i.numBytes(), {dim, n_heads * head_size}, {}});

        auto rms_ffn = weights.rms_ffn_weight.slice(i);
        tensors.push_back({pfx + "post_attention_layernorm.weight",
                            rms_ffn.data(), rms_ffn.numBytes(), {dim}, {}});

        auto w1_i = weights.w1.slice(i);
        tensors.push_back({pfx + "mlp.gate_proj.weight",
                            w1_i.data(), w1_i.numBytes(), {hidden_dim, dim}, {}});

        auto w2_i = weights.w2.slice(i);
        tensors.push_back({pfx + "mlp.down_proj.weight",
                            w2_i.data(), w2_i.numBytes(), {dim, hidden_dim}, {}});

        auto w3_i = weights.w3.slice(i);
        tensors.push_back({pfx + "mlp.up_proj.weight",
                            w3_i.data(), w3_i.numBytes(), {hidden_dim, dim}, {}});
    }

    tensors.push_back({"model.norm.weight",
                        weights.rms_final_weight.data(),
                        weights.rms_final_weight.numBytes(),
                        {dim}, {}});
    tensors.push_back({"lm_head.weight",
                        weights.wcls.data(),
                        weights.wcls.numBytes(),
                        {vocab_size, dim}, {}});

    // Build JSON header
    auto shape_to_json = [](const std::vector<size_t> &shape) -> std::string {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(shape[i]);
        }
        return s + "]";
    };

    size_t data_offset = 0;
    std::string header = "{\"__metadata__\":{\"format\":\"pt\"}";
    for (const auto &t : tensors) {
        size_t end = data_offset + t.num_bytes;
        header += ",\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":" + shape_to_json(t.shape) +
                  ",\"data_offsets\":[" + std::to_string(data_offset) + "," + std::to_string(end) + "]}";
        data_offset = end;
    }
    header += "}";

    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
        std::cerr << "Couldn't open " << output_path << " for writing\n";
        std::exit(EXIT_FAILURE);
    }
    uint64_t header_size = static_cast<uint64_t>(header.size());
    file.write(reinterpret_cast<const char *>(&header_size), sizeof(uint64_t));
    file.write(header.data(), static_cast<std::streamsize>(header_size));
    for (const auto &t : tensors) {
        file.write(reinterpret_cast<const char *>(t.data), static_cast<std::streamsize>(t.num_bytes));
    }
    file.close();
    std::cout << "Saved " << output_path << "\n";
}

// Writes a HuggingFace-compatible config.json for LlamaForCausalLM.
void saveHFConfig(const std::string &output_path, const TransformerConfig &config) {
    std::ofstream file(output_path);
    if (!file) {
        std::cerr << "Couldn't open " << output_path << " for writing\n";
        std::exit(EXIT_FAILURE);
    }
    file << "{\n"
         << "  \"architectures\": [\"LlamaForCausalLM\"],\n"
         << "  \"bos_token_id\": 1,\n"
         << "  \"eos_token_id\": 2,\n"
         << "  \"hidden_act\": \"silu\",\n"
         << "  \"hidden_size\": " << config.dim << ",\n"
         << "  \"intermediate_size\": " << config.hidden_dim << ",\n"
         << "  \"max_position_embeddings\": " << config.seq_len << ",\n"
         << "  \"model_type\": \"llama\",\n"
         << "  \"num_attention_heads\": " << config.n_heads << ",\n"
         << "  \"num_hidden_layers\": " << config.n_layers << ",\n"
         << "  \"num_key_value_heads\": " << config.n_kv_heads << ",\n"
         << "  \"rms_norm_eps\": 1e-5,\n"
         << "  \"tie_word_embeddings\": false,\n"
         << "  \"torch_dtype\": \"float32\",\n"
         << "  \"vocab_size\": " << std::abs(config.vocab_size) << "\n"
         << "}\n";
    file.close();
    std::cout << "Saved " << output_path << "\n";
}

// Exports model weights and config to a HuggingFace-compatible directory.
// Creates: {output_dir}/model.safetensors and {output_dir}/config.json
void exportHF(const std::string &output_dir,
              const TransformerConfig &config,
              const TransformerWeights<CPU, float32_t> &weights) {
    std::filesystem::create_directories(output_dir);
    saveSafetensors(output_dir + "/model.safetensors", config, weights);
    saveHFConfig(output_dir + "/config.json", config);
}


}  // namespace llama2cpp
#endif
#ifndef LLAMA2CPP_TRANSFORMER_HPP
#define LLAMA2CPP_TRANSFORMER_HPP
#include <string>
#include <cstdlib>
#include <memory>
#include <llama2cpp/ops.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace llama2cpp
{

    /**
     * @brief Transformer configuration
     *
     */
    struct TransformerConfig
    {
        int dim;        // transformer dimension
        int hidden_dim; // for ffn layers
        int n_layers;   // number of layers
        int n_heads;    // number of query heads
        int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        int vocab_size; // vocabulary size, usually 256 (byte-level)
        int seq_len;    // max sequence length
    };

    struct TransformerWeights
    {
        // token embedding table
        float *token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        float *rms_att_weight; // (layer, dim) rmsnorm weights
        float *rms_ffn_weight; // (layer, dim)
        // weights for matmuls. note dim == n_heads * head_size
        float *wq; // (layer, dim, n_heads * head_size)
        float *wk; // (layer, dim, n_kv_heads * head_size)
        float *wv; // (layer, dim, n_kv_heads * head_size)
        float *wo; // (layer, n_heads * head_size, dim)
        // weights for ffn
        float *w1; // (layer, hidden_dim, dim)
        float *w2; // (layer, dim, hidden_dim)
        float *w3; // (layer, hidden_dim, dim)
        // final rmsnorm
        float *rms_final_weight; // (dim,)
        // (optional) classifier weights for the logits, on the last layer
        float *wcls;
    };

    struct RunState
    {
        // current wave of activations
        float *x;      // activation at current time stamp (dim,)
        float *xb;     // same, but inside a residual branch (dim,)
        float *xb2;    // an additional buffer just for convenience (dim,)
        float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
        float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
        float *q;      // query (dim,)
        float *k;      // key (dim,)
        float *v;      // value (dim,)
        float *att;    // buffer for scores/attention values (n_heads, seq_len)
        float *logits; // output logits
        // kv cache
        float *key_cache;   // (layer, seq_len, dim)
        float *value_cache; // (layer, seq_len, dim)
    };

    void memory_map_weights(TransformerWeights *w, TransformerConfig *p, float *ptr, int shared_weights)
    {
        int head_size = p->dim / p->n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        unsigned long long n_layers = p->n_layers;
        w->token_embedding_table = ptr;
        ptr += p->vocab_size * p->dim;
        w->rms_att_weight = ptr;
        ptr += n_layers * p->dim;
        w->wq = ptr;
        ptr += n_layers * p->dim * (p->n_heads * head_size);
        w->wk = ptr;
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
        w->wv = ptr;
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
        w->wo = ptr;
        ptr += n_layers * (p->n_heads * head_size) * p->dim;
        w->rms_ffn_weight = ptr;
        ptr += n_layers * p->dim;
        w->w1 = ptr;
        ptr += n_layers * p->dim * p->hidden_dim;
        w->w2 = ptr;
        ptr += n_layers * p->hidden_dim * p->dim;
        w->w3 = ptr;
        ptr += n_layers * p->dim * p->hidden_dim;
        w->rms_final_weight = ptr;
        ptr += p->dim;
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        w->wcls = shared_weights ? w->token_embedding_table : ptr;
    }

    void read_checkpoint(const char *checkpoint, TransformerConfig *config, TransformerWeights *weights,
                         int *fd, float **data, ssize_t *file_size)
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file)
        {
            fprintf(stderr, "Couldn't open file %s\n", checkpoint);
            exit(EXIT_FAILURE);
        }
        // read in the config header
        if (fread(config, sizeof(TransformerConfig), 1, file) != 1)
        {
            exit(EXIT_FAILURE);
        }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config->vocab_size > 0 ? 1 : 0;
        config->vocab_size = abs(config->vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        *fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (*fd == -1)
        {
            fprintf(stderr, "open failed!\n");
            exit(EXIT_FAILURE);
        }
        *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED)
        {
            fprintf(stderr, "mmap failed!\n");
            exit(EXIT_FAILURE);
        }
        float *weights_ptr = *data + sizeof(TransformerConfig) / sizeof(float);
        memory_map_weights(weights, config, weights_ptr, shared_weights);
    }

    void malloc_run_state(RunState *s, TransformerConfig *p)
    {
        // we calloc instead of malloc to keep valgrind happy
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        s->x = (float *)calloc(p->dim, sizeof(float));
        s->xb = (float *)calloc(p->dim, sizeof(float));
        s->xb2 = (float *)calloc(p->dim, sizeof(float));
        s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
        s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
        s->q = (float *)calloc(p->dim, sizeof(float));
        s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
        s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
        s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
        s->logits = (float *)calloc(p->vocab_size, sizeof(float));
        // ensure all mallocs went fine
        if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits)
        {
            fprintf(stderr, "malloc failed!\n");
            exit(EXIT_FAILURE);
        }
    }

    void free_run_state(RunState *s)
    {
        free(s->x);
        free(s->xb);
        free(s->xb2);
        free(s->hb);
        free(s->hb2);
        free(s->q);
        free(s->att);
        free(s->logits);
        free(s->key_cache);
        free(s->value_cache);
    }

    class Transformer
    {
    public:
        using ptr = std::unique_ptr<Transformer>;
        Transformer(const std::string &checkpoint_path)
        {
            // read in the Config and the Weights from the checkpoint
            read_checkpoint(checkpoint_path.c_str(), &config, &weights, &fd, &data, &file_size);
            // allocate the RunState buffers
            malloc_run_state(&state, &config);
        }

        ~Transformer()
        {
            // close the memory mapping
            if (data != MAP_FAILED)
            {
                munmap(data, file_size);
            }
            if (fd != -1)
            {
                close(fd);
            }
            // free the RunState buffers
            free_run_state(&state);
        }

        auto forward(int token, int pos) -> float32_t *
        {
            // a few convenience variables
            TransformerConfig *p = &config;
            TransformerWeights *w = &weights;
            RunState *s = &state;
            float *x = s->x;
            int dim = p->dim;
            int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
            int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
            int hidden_dim = p->hidden_dim;
            int head_size = dim / p->n_heads;

            // copy the token embedding into x
            float *content_row = w->token_embedding_table + token * dim;
            memcpy(x, content_row, dim * sizeof(*x));

            // forward all the layers
            for (unsigned long long l = 0; l < p->n_layers; l++)
            {

                // attention rmsnorm
                rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

                // key and value point to the kv cache
                int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
                s->k = s->key_cache + loff + pos * kv_dim;
                s->v = s->value_cache + loff + pos * kv_dim;

                // qkv matmuls for this position
                matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
                matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
                matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

                // RoPE relative positional encoding: complex-valued rotate q and k in each head
                for (int i = 0; i < dim; i += 2)
                {
                    int head_dim = i % head_size;
                    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int v = 0; v < rotn; v++)
                    {
                        float *vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                        float v0 = vec[i];
                        float v1 = vec[i + 1];
                        vec[i] = v0 * fcr - v1 * fci;
                        vec[i + 1] = v0 * fci + v1 * fcr;
                    }
                }

                // multihead attention. iterate over all heads
                int h;
#pragma omp parallel for private(h)
                for (h = 0; h < p->n_heads; h++)
                {
                    // get the query vector for this head
                    float *q = s->q + h * head_size;
                    // attention scores for this head
                    float *att = s->att + h * p->seq_len;
                    // iterate over all timesteps, including the current one
                    for (int t = 0; t <= pos; t++)
                    {
                        // get the key vector for this head and at this timestep
                        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                        // calculate the attention score as the dot product of q and k
                        float score = 0.0f;
                        for (int i = 0; i < head_size; i++)
                        {
                            score += q[i] * k[i];
                        }
                        score /= sqrtf(head_size);
                        // save the score to the attention buffer
                        att[t] = score;
                    }

                    // softmax the scores to get attention weights, from 0..pos inclusively
                    softmax(att, pos + 1);

                    // weighted sum of the values, store back into xb
                    float *xb = s->xb + h * head_size;
                    memset(xb, 0, head_size * sizeof(float));
                    for (int t = 0; t <= pos; t++)
                    {
                        // get the value vector for this head and at this timestep
                        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                        // get the attention weight for this timestep
                        float a = att[t];
                        // accumulate the weighted value into xb
                        for (int i = 0; i < head_size; i++)
                        {
                            xb[i] += a * v[i];
                        }
                    }
                }

                // final matmul to get the output of the attention
                matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

                // residual connection back into x
                for (int i = 0; i < dim; i++)
                {
                    x[i] += s->xb2[i];
                }

                // ffn rmsnorm
                rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

                // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
                // first calculate self.w1(x) and self.w3(x)
                matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
                matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

                // SwiGLU non-linearity
                for (int i = 0; i < hidden_dim; i++)
                {
                    float val = s->hb[i];
                    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                    val *= (1.0f / (1.0f + expf(-val)));
                    // elementwise multiply with w3(x)
                    val *= s->hb2[i];
                    s->hb[i] = val;
                }

                // final matmul to get the output of the ffn
                matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

                // residual connection
                for (int i = 0; i < dim; i++)
                {
                    x[i] += s->xb[i];
                }
            }

            // final rmsnorm
            rmsnorm(x, x, w->rms_final_weight, dim);

            // classifier into logits
            matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
            return s->logits;
        }

        auto getConfig() const -> const TransformerConfig&
        {
            return config;
        }

    private:
        TransformerConfig config;   // the hyperparameters of the architecture (the blueprint)
        TransformerWeights weights; // the weights of the model
        RunState state;             // buffers for the "wave" of activations in the forward pass
        // some more state needed to properly clean up the memory mapping (sigh)
        int fd;            // file descriptor for memory mapping
        float *data;       // memory mapped data pointer
        ssize_t file_size; // size of the checkpoint file in bytes
    };

}
#endif
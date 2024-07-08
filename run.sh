CUDA_VISIBLE_DEVICES=0,1,2,3 
python run_quarot.py \
        --model Mixtral-8x7B-v0.1 \
        --model-path /share/projset/hxs-6k/huangxiusheng/AMD/model_saves/mistralai/Mixtral-8x7B-v0.1 \
        --rotate \
        --w-bits 4 \
        --w-gptq \
        --a-bits 4 \
        --k-bits 4 \
        --v-bits 4 \
        --lm-eval \
        --tasks piqa \
        --device cuda:0 \
        --no-wandb \
        --hf-token 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
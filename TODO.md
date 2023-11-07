* update README with different scenarios (full training, lora, pseudo qlora, pre-training)
* make logging_steps, eval_steps, save_steps arguments for train
* make save_total_limit, load_best_model_at_end arguments for train
* make test_size an argument for train
* add target_dir to snapshot and readjust metadata for Llama-2-7b-hf
* add replay functionality
* implement early stopping, use load_best_model_at_end?
* distributed training on multiple GPUs with speedup - torchrun.distributed/accelerate/deepspeed
* train directly from adapter with automated detection and merging of base model and previous adapters
* print and compare tensors for base models, quantized models, (q)lora adapters, and merged models
* fix training with gptq model (qlora)
* see if other layers can be meaningfully LoRA'ed
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* quantization-aware training (llm-qat)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not needed
* more parameters (e.g. LoRA parameters, training parameters)
* implement test suite

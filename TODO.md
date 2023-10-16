* test whether trained models actually perform better after merging (lora)
* test whether trained models actually perform better after merging and quantization (pseudo qlora)
* update README with different scenarios (lora, pseudo qlora)
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not nee3ded
* move parsing of arguments into utils (by name)
* more parameters (e.g. LoRA size)
* implement test suite
* integrate snapshot/download functionality

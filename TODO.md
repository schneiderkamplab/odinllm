* update README with different scenarios (lora, pseudo qlora)
* implement arbitrary datasets for at least two formats (alpaca, raw) for train
* quantize adapter weights and merge with quantized model (real qlora)
* quantized adapter and merge with quantized model (qalora)
* figure out device maps such that CUDA_VISIBLE_DEVICES is not nee3ded
* move parsing of arguments into utils (by name)
* more parameters (e.g. LoRA size)
* implement test suite

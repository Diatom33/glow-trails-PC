# glow-trails-PC
PC reimplementation of the FPGA video processing in [https://github.com/kiran-vuksanaj/glow-trails](https://github.com/kiran-vuksanaj/glow-trails). This is intended to make glowy spinning arts props look cooler.

Takes webcam input and outputs a processed video stream, where each pixel above a certain brightness threshold is maintained into the next frame with some decay factor, while other pixels are replaced with the next frame.

May require parameter tuning. In its current state, requires a CUDA-enabled GPU.

# glow-trails-PC
Linux reimplementation of the FPGA video processing in [kiran-vuksanaj/glow-trails](https://github.com/kiran-vuksanaj/glow-trails). This is intended to make glowy spinning arts props look cooler. Both repos implement a very similar transformation to [Drex's ghost trails](https://www.youtube.com/watch?v=ZhylHYvUFT4), but in real time as opposed to being purely an after effect.

Takes webcam input and outputs a processed video stream, where pixels above a certain brightness threshold genrerate glow trails.

May require parameter tuning for your specific use case. May require tuning a lot of other stuff, frankly, I made this to work for my own setup and haven't made it maximally customizable yet. In its current state, requires a CUDA-enabled GPU.

## Usage

```bash
python main.py --threshold 0.3 --decay 0.95 --mirror True --width 1280 --height 720 --export ~/out.avi
```

Options:
- `--threshold`: Threshold for the brightness of a pixel to be maintained into the next frame (default: 0.3).
- `--decay`: Decay factor for the brightness of a pixel to be maintained into the next frame (default: 0.95).
- `--mirror`: Whether to mirror the video output horizontally (default: True).
- `--width`: Width of the video stream (default: 1280).
- `--height`: Height of the video stream (default: 720).
- `--export`: Path to save the output video to (default: None).

Height and width must match one of the resolutions of the webcam. (findable with `v4l2-ctl --list-formats-ext -d /dev/video0`)

Allows a wide range of export formats, it probably has the one you need. Hasn't been extensively tested.

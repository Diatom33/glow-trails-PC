import torch
import cv2
import time
import tkinter as tk
import argparse
import sys
import os
import numpy as np
from screeninfo import get_monitors

# Default configuration values
DEFAULT_THRESHOLD = 0.3
DEFAULT_DECAY = 0.95
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

def str_to_bool(value: str) -> bool:
    """
    Convert a string representation of a boolean to a boolean value.
    """
    return value.lower() == 'true'

class GlowTrails:
    def __init__(
        self, 
        threshold: float = DEFAULT_THRESHOLD,
        decay: float = DEFAULT_DECAY,
        camera_width: int = DEFAULT_WIDTH,
        camera_height: int = DEFAULT_HEIGHT,
        mirror: bool = True,
        export: bool = False,
        export_fps: str = 'average',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.threshold = threshold
        self.decay = decay
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.device = device
        self.mirror = mirror
        self.export = export
        self.export_fps = export_fps
        self.video_writer = None
        self.frames_buffer = []  # Buffer to store frames for initial FPS calculation
        self.initial_frames = 30  # Number of frames to calculate average FPS before initializing VideoWriter
        
        # Initialize the webcam with a specific backend to avoid GStreamer warning
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Using V4L2 backend
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set pixel format to MJPG
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if not self.cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            print("Warning: Unable to set MJPG format. The webcam might not support it.")
        else:
            print("MJPG format set successfully.")

        # Set webcam resolution based on user input
        if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width):
            print(f"Warning: Unable to set frame width to {self.camera_width}.")
        if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height):
            print(f"Warning: Unable to set frame height to {self.camera_height}.")
        
        # Set webcam's FPS
        if not self.cap.set(cv2.CAP_PROP_FPS, 30):
            print("Warning: Unable to set FPS to 30.")
        
        # Retrieve and verify pixel format
        current_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((current_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"Current FOURCC code: {fourcc_str}")
        if fourcc_str != 'MJPG':
            print("Warning: The webcam is not using MJPG format.")
        
        self.webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.webcam_fps <= 0:
            # Fallback to common default if unable to get FPS
            self.webcam_fps = 30.0
            print(f"Warning: Could not determine webcam FPS, assuming {self.webcam_fps}")
        else:
            print(f"Webcam FPS: {self.webcam_fps:.1f}")
        
        print(f"Using device: {self.device}")

        # Retrieve screen dimensions
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.destroy()

        # Initialize current monitor resolution
        self.update_current_monitor_resolution()

        # Create a named window with WINDOW_NORMAL to allow resizing
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Initialize VideoWriter if export is enabled
        if self.export:
            if isinstance(self.export, str):
                # Delay initialization of VideoWriter until average FPS is calculated
                print("Export is enabled. VideoWriter will initialize after calculating average FPS.")
            else:
                print("Warning: Export parameter should be a file path or 'False'. Export disabled.")
                self.export = False

        self.video_writer_initialized = False  # Flag to check if VideoWriter is initialized

    def update_current_monitor_resolution(self):
        """
        Update the current monitor's resolution based on the window's position.
        """
        # Attempt to get the window's current position
        window_pos = self.get_window_position()
        if window_pos is None:
            # Fallback to primary monitor if unable to get window position
            monitor = get_monitors()[0]
        else:
            x, y = window_pos
            # Find the monitor that contains the window's position
            monitor = next(
                (m for m in get_monitors() if m.x <= x < m.x + m.width and m.y <= y < m.y + m.height),
                get_monitors()[0]
            )
        
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        print(f"Current Monitor Resolution: {self.screen_width}x{self.screen_height}")

    def get_window_position(self):
        """
        Get the current position of the OpenCV window.
        Note: OpenCV does not provide a direct method to get window position.
        This function uses a workaround for Windows using pygetwindow.
        """
        try:
            import subprocess
            # Use wmctrl to list all windows with geometry
            output = subprocess.check_output(['wmctrl', '-lG']).decode('utf-8')
            for line in output.splitlines():
                parts = line.split()
                x = int(parts[2])
                y = int(parts[3])
                title = ' '.join(parts[7:])
                
                if 'Output' in title:
                    return (x, y)
            return None
        except Exception as e:
            print(f"Warning: Unable to get window position due to: {e}")
            return None

    def initialize_video_writer(self, avg_fps: float):
        """
        Initialize the VideoWriter with the appropriate codec based on file extension.
        """
        # Mapping of file extensions to FOURCC codes
        extension_fourcc_map = {
            '.avi': 'MJPG',
            '.mp4': 'mp4v',
            '.mkv': 'X264',
            '.mov': 'avc1',
            # Add more mappings as needed
        }

        export_path = self.export
        # Extract file extension
        _, ext = os.path.splitext(export_path) #type: ignore
        ext = ext.lower()

        # Get the corresponding FOURCC code
        fourcc_code = extension_fourcc_map.get(ext)
        if not fourcc_code:
            print(f"Warning: Unsupported file extension '{ext}'. Export disabled.")
            return

        fourcc_out = cv2.VideoWriter_fourcc(*fourcc_code)
        try:
            # Define the codec and create VideoWriter object with chosen FPS
            chosen_fps = avg_fps if self.export_fps == 'average' else self.webcam_fps
            self.video_writer = cv2.VideoWriter(
                export_path, 
                fourcc_out, 
                chosen_fps, 
                (self.camera_width, self.camera_height)
            )
            if not self.video_writer.isOpened():
                print(f"Warning: Unable to open video file for writing at {export_path}. Export disabled.")
                self.video_writer = None
            else:
                fps_type = "average" if self.export_fps == 'average' else "webcam"
                print(f"Video output will be saved to {export_path} with codec '{fourcc_code}' at {fps_type} FPS ({chosen_fps:.2f}).")
                self.video_writer_initialized = True
        except Exception as e:
            print(f"Error initializing video writer: {e}. Export disabled.")
            self.video_writer = None

    def get_luminance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Calculate the luminance of the image.
        Image tensor shape: (C, H, W)
        Returns: (H, W)
        """
        # Vectorized luminance calculation using torch operations
        return (0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2])

    def transform(self, image_current: torch.Tensor, image_next: torch.Tensor) -> torch.Tensor:
        """
        Apply threshold-based transformation between two images.
        Both images are expected to be on the same device.
        """
        with torch.no_grad():  # Disable gradient calculations for inference
            luminance_next = self.get_luminance(image_next)
            mask = luminance_next > self.threshold  # Shape: (H, W)
            mask = mask.unsqueeze(0)  # Shape: (1, H, W)
            # Apply weighted average for new trails with 80:20 new:old ratio
            out = image_current * (~mask) * self.decay + (0.25 * image_current + 0.75 * image_next) * mask
        return out

    def get_inputs(self) -> torch.Tensor:
        """
        Capture a frame from the webcam and return it as a CUDA tensor.
        Tensor shape: (C, H, W)
        """
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Failed to capture image from webcam")

        # Convert the image from BGR to RGB and resize if necessary
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor and normalize to [0, 1]
        tensor = torch.from_numpy(frame_rgb).float() / 255.0  # Shape: (H, W, C)

        # Rearrange dimensions to (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        # Move to CUDA
        tensor = tensor.to(self.device, non_blocking=True)

        return tensor

    def send_outputs(self, tensor: torch.Tensor):
        """
        Display the tensor as an image on the screen with padding to match screen size.
        Tensor shape: (C, H, W)
        """
        # Move tensor to CPU and convert to NumPy array
        frame = tensor.cpu().detach().numpy()

        # Rearrange dimensions back to (H, W, C)
        frame = frame.transpose(1, 2, 0)

        # Convert from RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Denormalize the image from [0, 1] to [0, 255] and convert to uint8
        frame_bgr_uint8 = (frame_bgr * 255).astype('uint8')

        # **Mirror the video output horizontally**
        if self.mirror:
            frame_bgr_uint8 = cv2.flip(frame_bgr_uint8, 1)  # 1 denotes horizontal flipping

        # **Write the processed frame to the video file if exporting is enabled and initialized**
        if self.video_writer:
            self.video_writer.write(frame_bgr_uint8)

        # **Resize the frame while maintaining aspect ratio for display**
        frame_height, frame_width = frame_bgr_uint8.shape[:2]
        screen_aspect = self.screen_width / self.screen_height
        frame_aspect = frame_width / frame_height

        if frame_aspect > screen_aspect:
            # Frame is wider than screen
            new_width = self.screen_width
            new_height = int(new_width / frame_aspect)
        else:
            # Frame is taller than screen
            new_height = self.screen_height
            new_width = int(new_height * frame_aspect)

        frame_resized = cv2.resize(
            frame_bgr_uint8, 
            (new_width, new_height), 
            interpolation=cv2.INTER_AREA
        )

        # **Create a black background image for padding**
        frame_display = np.zeros((self.screen_height, self.screen_width, 3), dtype='uint8')

        # **Calculate top-left corner for centered placement**
        x_offset = (self.screen_width - new_width) // 2
        y_offset = (self.screen_height - new_height) // 2

        # **Place the resized frame onto the black background**
        frame_display[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_resized

        # **Display the padded image**
        cv2.imshow('Output', frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def process_stream(self):
        """
        Continuously capture, process, and display video frames.
        """
        try:
            image_current = self.get_inputs()
            frame_count = 0
            start_time_avg = time.time()
            last_check_time = time.time()
            check_interval = 1  # seconds
            
            while True:
                frame_start_time = time.time()
                image_next = self.get_inputs()
                transformed = self.transform(image_current, image_next)
                self.send_outputs(transformed)
                image_current = transformed
                
                # Calculate instantaneous FPS
                frame_time = time.time() - frame_start_time
                instant_fps = 1 / frame_time if frame_time > 0 else 0
                
                # Calculate average FPS
                elapsed_time = time.time() - start_time_avg
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Calculate FPS ratio (actual/webcam)
                fps_ratio = avg_fps / self.webcam_fps if self.webcam_fps > 0 else 0
                
                print(f"FPS: {instant_fps:5.1f} | Avg FPS: {avg_fps:5.1f} | "
                      f"Ratio to webcam ({self.webcam_fps:.1f}): {fps_ratio:4.2f}")
                
                frame_count += 1

                # Start of Selection
                # Buffer frames to calculate average FPS before initializing VideoWriter
                if self.export and not self.video_writer_initialized:
                    frame_bgr_uint8 = transformed.cpu().numpy().astype('uint8')
                    frame_bgr_uint8 = cv2.cvtColor(frame_bgr_uint8, cv2.COLOR_RGB2BGR)
                    self.frames_buffer.append(frame_bgr_uint8)
                    if frame_count >= self.initial_frames:
                        avg_fps_calculated = frame_count / elapsed_time
                        self.initialize_video_writer(avg_fps_calculated)
                        # Write buffered frames
                        if self.video_writer:
                            for buffered_frame in self.frames_buffer:
                                self.video_writer.write(buffered_frame)
                        self.frames_buffer = []  # Clear buffer
                if self.video_writer:
                    for buffered_frame in self.frames_buffer:
                        self.video_writer.write(buffered_frame)
                self.frames_buffer = []  # Clear buffer

                # Periodically check for screen resolution changes
                current_time = time.time()
                if current_time - last_check_time > check_interval:
                    self.update_current_monitor_resolution()
                    last_check_time = current_time
                
        except KeyboardInterrupt:
            print("Interrupted by user.")
            self.stop()
        except Exception as e:
            print(f"An error occurred: {e}")
            self.stop()

    def stop(self):
        """
        Release resources and close windows.
        """
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
            print(f"Video file saved to {self.export}.")
        cv2.destroyAllWindows()
        print("Resources released. Exiting.")

def parse_arguments():
    """
    Parse and validate command-line arguments.
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="GlowTrails Video Processor")
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=DEFAULT_THRESHOLD,
        help=f"Threshold for luminance (0-1). Default is {DEFAULT_THRESHOLD}."
    )
    parser.add_argument(
        '--decay', 
        type=float, 
        default=DEFAULT_DECAY,
        help=f"Decay factor for glow trails. Default is {DEFAULT_DECAY}."
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=DEFAULT_WIDTH,
        help=f"Camera resolution width in pixels. Default is {DEFAULT_WIDTH}."
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=DEFAULT_HEIGHT,
        help=f"Camera resolution height in pixels. Default is {DEFAULT_HEIGHT}."
    )
    parser.add_argument(
        '--mirror',
        type=str,
        default='True',
        help="Mirror the video output horizontally. Default is True."
    )
    # New argument for exporting video
    parser.add_argument(
        '--export',
        type=str,
        default='False',
        help="Export the video output to a file. Provide the file path with desired extension or 'False' to disable. Default is False."
    )
    # New argument for export FPS option
    parser.add_argument(
        '--export-fps',
        type=str,
        choices=['average', 'webcam'],
        default='average',
        help="Choose the FPS for the exported video: 'average' to use the calculated average FPS or 'webcam' to use the webcam's FPS. Default is 'average'."
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("Threshold must be between 0 and 1.")
    
    # Validate decay
    if not (0.0 < args.decay < 1.0):
        parser.error("Decay must be a float between 0 and 1 (non-inclusive).")
    
    # Validate width and height
    if args.width <= 0:
        parser.error("Width must be a positive integer.")
    if args.height <= 0:
        parser.error("Height must be a positive integer.")
    
    # Validate mirror
    if args.mirror.lower() not in ['true', 'false']:
        parser.error("Mirror must be a boolean value (True or False).")
    
    # Validate export
    if args.export.lower() in ['true', 'false']:
        export = str_to_bool(args.export)
    else:
        # Assume it's a file path
        export = args.export
    
    args.export = export
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        processor = GlowTrails(
            threshold=args.threshold,
            decay=args.decay,
            camera_width=args.width,
            camera_height=args.height,
            mirror=str_to_bool(args.mirror),
            export=args.export,
            export_fps=args.export_fps  # Pass the new export_fps argument
        )
        processor.process_stream()
    except Exception as e:
        print(f"Failed to start GlowTrails: {e}")
        sys.exit(1)

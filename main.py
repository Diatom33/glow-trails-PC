import torch
import cv2
import time
import tkinter as tk
import argparse
import sys

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
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.threshold = threshold
        self.decay = decay
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.device = device
        self.mirror = mirror
        self.export = export
        self.video_writer = None
        
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
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        print(f"Screen Resolution: {self.screen_width}x{self.screen_height}")

        # Create a named window with WINDOW_NORMAL to allow resizing
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Initialize VideoWriter if export is enabled
        if self.export and isinstance(self.export, str):
            try:
                # Define the codec and create VideoWriter object
                fourcc_out = cv2.VideoWriter_fourcc(*'MJPG')
                self.video_writer = cv2.VideoWriter(
                    self.export, 
                    fourcc_out, 
                    self.webcam_fps, 
                    (self.screen_width, self.screen_height)
                )
                if not self.video_writer.isOpened():
                    print(f"Warning: Unable to open video file for writing at {self.export}. Export disabled.")
                    self.video_writer = None
                else:
                    print(f"Video output will be saved to {self.export}.")
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
            out = image_current * (~mask) + (image_current * 0.6 + image_next * 0.4) * mask
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
        Display the tensor as an image on the screen.
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

        # **Resize the frame to fit the screen**
        frame_resized = cv2.resize(
            frame_bgr_uint8, 
            (self.screen_width, self.screen_height), 
            interpolation=cv2.INTER_AREA
        )

        # Display the image
        cv2.imshow('Output', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()
        
        # Write the frame to the video file if exporting is enabled
        if self.video_writer:
            self.video_writer.write(frame_resized)

    def process_stream(self):
        """
        Continuously capture, process, and display video frames.
        """
        try:
            image_current = self.get_inputs()
            frame_count = 0
            start_time_avg = time.time()
            
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
        help="Export the video output to a file. Provide the file path or 'False' to disable. Default is False."
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
    if args.export.lower() not in ['true', 'false']:
        # If not strictly 'true' or 'false', treat as file path
        export = args.export
    else:
        export = str_to_bool(args.export)
    
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
            export=args.export
        )
        processor.process_stream()
    except Exception as e:
        print(f"Failed to start GlowTrails: {e}")
        sys.exit(1)

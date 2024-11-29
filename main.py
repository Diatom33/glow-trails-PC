import torch
import cv2
import time

# Adjusted threshold to match normalized scale (0-1)
threshold = 0.85
decay = 0.999

class GlowTrails:
    def __init__(
        self, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.device = device
        # Initialize the webcam with a specific backend to avoid GStreamer warning
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Using V4L2 backend
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Get webcam's native FPS
        self.webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.webcam_fps <= 0:
            # Fallback to common default if unable to get FPS
            self.webcam_fps = 30.0
            print(f"Warning: Could not determine webcam FPS, assuming {self.webcam_fps}")
        else:
            print(f"Webcam FPS: {self.webcam_fps:.1f}")
        
        print(f"Using device: {self.device}")

    def get_luminance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Calculate the luminance of the image.
        Image tensor shape: (C, H, W)
        Returns: (H, W)
        """
        # Calculate luminance using standard coefficients
        r, g, b = image[0], image[1], image[2]  # RGB channels
        return 0.2126*r + 0.7152*g + 0.0722*b

    def transform(self, image_current: torch.Tensor, image_next: torch.Tensor) -> torch.Tensor:
        """
        Apply threshold-based transformation between two images.
        Both images are expected to be on the same device.
        """
        with torch.no_grad():  # Disable gradient calculations for inference
            mask = self.get_luminance(image_current) > threshold  # Shape: (H, W)
            mask = mask.unsqueeze(0).float()  # Shape: (1, H, W)
            out = decay*image_current * mask + image_next * (1 - mask)
        return out

    def get_inputs(self) -> torch.Tensor:
        """
        Capture a frame from the webcam and return it as a CUDA tensor.
        Tensor shape: (C, H, W)
        """
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("Failed to capture image from webcam")

        # Convert the image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to torch tensor and normalize to [0, 1]
        tensor = torch.from_numpy(frame_rgb).float() / 255.0  # Shape: (H, W, C)

        # Rearrange dimensions to (C, H, W)
        tensor = tensor.permute(2, 0, 1)

        # Move to CUDA
        tensor = tensor.to(self.device)

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

        # Display the image
        cv2.imshow('Output', frame_bgr_uint8)
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
            
            while True:
                frame_start_time = time.time()
                image_next = self.get_inputs()
                transformed = self.transform(image_current, image_next)
                self.send_outputs(transformed)
                image_current = transformed
                
                # Calculate instantaneous FPS
                frame_time = time.time() - frame_start_time
                instant_fps = 1 / frame_time
                
                # Calculate average FPS
                frame_count += 1
                avg_fps = frame_count / (time.time() - start_time_avg)
                
                # Calculate FPS ratio (actual/webcam)
                fps_ratio = avg_fps / self.webcam_fps
                
                print(f"FPS: {instant_fps:5.1f} | Avg FPS: {avg_fps:5.1f} | "
                      f"Ratio to webcam ({self.webcam_fps:.1f}): {fps_ratio:4.2f}")
                
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
        cv2.destroyAllWindows()
        print("Resources released. Exiting.")

if __name__ == "__main__":
    processor = GlowTrails()
    processor.process_stream()

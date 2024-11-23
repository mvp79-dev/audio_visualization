import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, VideoClip
from pathlib import Path
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from tempfile import NamedTemporaryFile
from fastapi.responses import StreamingResponse, FileResponse
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

class AnimatedAudioVisualizer:
    def __init__(self, visualization_type='waveform', fps=30):
        self.visualization_type = visualization_type
        self.fps = fps

    def make_frame(self, t, y, sr, total_duration):
        """Generate the frame for time `t`."""
        fig, (ax_waveform, ax_spectrogram) = plt.subplots(1, 2, figsize=(18, 5))

        # Set background to black
        fig.patch.set_facecolor('black')  # Black background for the figure
        ax_waveform.set_facecolor('black')  # Black background for the waveform axis
        ax_spectrogram.set_facecolor('black')  # Black background for the spectrogram axis

        # Determine the segment of the audio to plot based on time `t`
        samples_per_frame = int(sr / self.fps)
        current_sample = int(t * sr)
        start = max(0, current_sample - samples_per_frame // 2)
        end = min(len(y), current_sample + samples_per_frame // 2)

        if self.visualization_type in ['waveform', 'both']:
            # Plot the waveform segment
            ax_waveform.plot(np.arange(start, end), y[start:end], color='blue')
            ax_waveform.set_ylim(-1, 1)  # Set consistent y-axis limits
            ax_waveform.set_xlim(start, end)
            ax_waveform.set_title("Waveform")
            ax_waveform.axis('off')  # Hide axes for a clean visualization

        if self.visualization_type in ['spectrogram', 'both']:
            # Compute the spectrogram using librosa
            hop_length = 512  # Controls the overlap in STFT
            n_fft = 2048  # FFT window size
            D = librosa.amplitude_to_db(librosa.stft(y[start:end], n_fft=n_fft, hop_length=hop_length), ref=np.max)

            # Plot the spectrogram as an image
            img = ax_spectrogram.imshow(D, aspect='auto', origin='lower', cmap='inferno', extent=[start, end, 0, sr // 2])
            ax_spectrogram.set_title("Spectrogram")
            ax_spectrogram.axis('off')  # Hide axes for a clean visualization

        # Save the frame to a numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return frame

    def create_visualization(self, audio_path, output_path):
        """Create animated audio visualization video."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = librosa.get_duration(y=y, sr=sr)

            # Create VideoClip with the make_frame function
            clip = VideoClip(lambda t: self.make_frame(t, y, sr, total_duration), duration=total_duration)
            clip = clip.set_fps(self.fps)

            # Attach audio
            audio_clip = AudioFileClip(audio_path)
            video = clip.set_audio(audio_clip)

            # Write to file
            video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise

@app.post("/upload/")
async def create_visualization(file: UploadFile = File(...), type: str = 'waveform', fps: int = 30):
    """API endpoint to process uploaded audio and return the visualized MP4 video."""
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Define the output folder
        output_folder = "output_folder"  # Set the path to the desired output directory
        os.makedirs(output_folder, exist_ok=True)

        # Set the output file path with the same name as the uploaded file
        output_path = os.path.join(output_folder, f"output_{file.filename}.mp4")

        # Call your function that processes the file
        visualizer = AnimatedAudioVisualizer(visualization_type=type, fps=fps)
        visualizer.create_visualization(temp_file_path, output_path)

        # Return the processed video
        return FileResponse(output_path, media_type="video/mp4")

    except Exception as e:
        logging.error(f"Failed to process file: {str(e)}")
        return {"error": f"Failed to process file: {str(e)}"}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

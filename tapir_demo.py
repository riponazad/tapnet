import haiku as hk
import jax
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import tree
from PIL import Image

import tapir_model
from utils import transforms
from utils import viz_utils


######
import cv2 as cv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Save frames from a video to a directory")
    parser.add_argument("--checkpoint", default="", help="Path to the input directory where images are stored")
    parser.add_argument("--video", default="", help="Path to the example video file")

    args = parser.parse_args()
    return args


def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = tapir_model.TAPIR(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs




#### Utility functions #####

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames


def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  return visibles

def inference(frames, query_points):
  """Inference on one video.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8
    query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

  Returns:
    tracks: [num_points, 3], [-1, 1], [t, y, x]
    visibles: [num_points, num_frames], bool
  """
  # Preprocess video to match model inputs format
  frames = preprocess_frames(frames)
  num_frames, height, width = frames.shape[0:3]
  query_points = query_points.astype(np.float32)
  frames, query_points = frames[None], query_points[None]  # Add batch dimension

  # Model inference
  rng = jax.random.PRNGKey(42)
  outputs, _ = model_apply(params, state, rng, frames, query_points)
  outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
  tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

  # Binarize occlusions
  visibles = postprocess_occlusions(occlusions, expected_dist)
  return tracks, visibles


def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points

def play_video(video_file):
    
    # Open the video file
    cap = cv.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, None
    
    # Get the resolution (width and height) of the video
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # List to store frames for the GIF
    frames = []
    new_frame = True

    while True:
        #Restart the video from the beginning
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            #Read a frame from the video
            ret, frame = cap.read()

            # Break the inner loop if we have reached the end of the video
            if not ret:
                new_frame = False
                break

            # Display the frame
            cv.imshow('Video Player', frame)

            # Append the frame to the list if it is new
            if new_frame == True:
               frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))

    
            #Check for user input to quit (press 'q' key)
            key = cv.waitKey(30)
            if key == ord('q'):
                # Release the video capture object and close the display window
                cap.release()
                cv.destroyAllWindows()

                # Save the video as a gif file
                output_gif_file = "outputs/input_video.gif" 
                frames[0].save(output_gif_file, save_all=True, append_images=frames[1:], loop=0)
                print(f"Input video is saved as a GIF file to {output_gif_file}")
                return width, height

def get_resized_frames(video_file, width, height):
    resized_frames = []

    # Open the video file
    cap = cv.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        # Resize the frame
        resized_frame = cv.resize(frame, (width, height))

        # Append the resized frame to the list
        #resized_frames.append(Image.fromarray(cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)))
        resized_frames.append(resized_frame)

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    resized_frames_np = np.array(resized_frames)

    return resized_frames_np


def visualize_video(frames_np, fps=30, video_file="output_video"):
    if frames_np is None or len(frames_np) == 0:
        print("No frames to visualize.")
        return

    height, width, _ = frames_np[0].shape

    # Create a window to display the video
    cv.namedWindow("Video Player", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video Player", width, height)

    # List to store frames for the GIF
    frames = []
    frame_cnt = 0

    while True:
        for frame in frames_np:
            cv.imshow('Video Player', frame)
            # Append the frame to the list if it is new
            if frame_cnt < frames_np.shape[0]:
               frame_cnt += 1
               frames.append(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)))
               #frames.append(frame)

            # Check for user input to quit (press 'q' key)
            key = cv.waitKey(int(1000 / fps))
            if key == ord('q'):
                cv.destroyAllWindows()
                # Save the output video as a gif file
                output_gif_file = f"outputs/{video_file}.gif"
                # Convert frames to PIL Image objects
                frames = [Image.fromarray(np.uint8(frame)) for frame in frames] 
                frames[0].save(output_gif_file, save_all=True, append_images=frames[1:], loop=0)
                print(f"Output video of random points is saved as a GIF file to {output_gif_file}")
                return
            
# Event handler for mouse clicks
def on_click(event):
    if event.button == 1 and event.inaxes == ax:  # Left mouse button clicked
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        select_points.append(np.array([x, y]))

        color = colormap[len(select_points) - 1]
        color = tuple(np.array(color) / 255.0)
        ax.plot(x, y, 'o', color=color, markersize=5)
        plt.draw()


def convert_select_points_to_query_points(frame, points):
  """Convert select points to query points.

  Args:
    points: [num_points, 2], in [x, y]
  Returns:
    query_points: [num_points, 3], in [t, y, x]
  """
  points = np.stack(points)
  query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
  query_points[:, 0] = frame
  query_points[:, 1] = points[:, 1]
  query_points[:, 2] = points[:, 0]
  return query_points


if __name__ == "__main__":
    args = parse_args()

    # Load checkpoint
    checkpoint_path = args.checkpoint
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    #Build model
    model = hk.transform_with_state(build_model)
    model_apply = jax.jit(model.apply)

    #Show the example video
    width, height = play_video(args.video)

    #Predict Sparse Point Tracks
    resize_height = 256  
    resize_width = 256 
    num_points = 20
    sampling_frame = 0

    frames = get_resized_frames(args.video, resize_width, resize_height)
    query_points = sample_random_points(sampling_frame, frames.shape[1], frames.shape[2], num_points)
    tracks, visibles = inference(frames, query_points)
    # print(frames.shape)

    # Visualize sparse point tracks
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
    frames = get_resized_frames(args.video, width, height)
    video_viz = viz_utils.paint_point_track(frames, tracks, visibles)
    # print(video_viz.shape)
    visualize_video(video_viz, fps=10, video_file="output_video1")


    #Select Any Points at Any Frame
    # Generate a colormap with 20 points, no need to change unless select more than 20 points
    colormap = viz_utils.get_colors(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    print(frames[sampling_frame].shape)
    ax.imshow(Image.fromarray(cv.cvtColor(frames[sampling_frame], cv.COLOR_BGR2RGB)))
    ax.axis('off')
    ax.set_title('You can select more than 1 points. After select enough points, close the window.')

    select_points = []

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    # Save the plot as an image file (e.g., PNG)
    output_image_file = 'outputs/selected_points.png'
    fig.savefig(output_image_file)
    print(f"Plot with selected points saved to {output_image_file}")
    print(select_points)

    frames = get_resized_frames(args.video, resize_width, resize_height)
    query_points = convert_select_points_to_query_points(sampling_frame, select_points)
    query_points = transforms.convert_grid_coordinates(
    query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')
    tracks, visibles = inference(frames, query_points)

    # Visualize sparse point tracks
    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
    frames = get_resized_frames(args.video, width, height)
    video_viz = viz_utils.paint_point_track(frames, tracks, visibles, colormap)
    visualize_video(video_viz, fps=10, video_file="output_video2")
    

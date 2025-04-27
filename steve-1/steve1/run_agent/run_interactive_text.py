import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading

WIDTH = '640'
HEIGHT = '360'
FRAMERATE = '20'
SHM_SOCKET = "/tmp/minecraft_video"
WINDOW_NAME="640x360_1mb.mp4 - VLC media player"
#WINDOW_NAME = "Minecraft* 1.19.4 - Singleplayer"
MODEL = 'foundation-model-2x.model'
WEIGHTS = 'rl-from-foundation-2x.weights'

# Initialize GStreamer
Gst.init(None)

# Function to handle user input
def user_input_thread():
    while True:
        user_input = input("Enter something: ")
        print(f"You entered: {user_input}")
        if user_input.lower() == "quit":
            print("Exiting user input thread...")
            break

# Function to handle incoming video frames
def video_frame_thread():
    # Create a GStreamer pipeline
    pipeline_description = f"""
        shmsrc socket-path={SHM_SOCKET} !
        video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/2 ! 
        appsink name=sink max-buffers=1 drop=True emit-signals=True
    """
    pipeline = Gst.parse_launch(pipeline_description)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Create a GLib main loop to handle GStreamer events
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Exiting video frame thread...")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    # Start the user input thread
    input_thread = threading.Thread(target=user_input_thread)
    input_thread.daemon = True
    input_thread.start()

    # Start the video frame thread
    video_thread = threading.Thread(target=video_frame_thread)
    video_thread.daemon = True
    video_thread.start()

    # Wait for both threads to finish
    input_thread.join()
    video_thread.join()

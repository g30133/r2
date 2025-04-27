# Variables
#WINDOW_NAME="640x360_1mb.mp4 - VLC media player"
WINDOW_NAME="Minecraft* 1.19.4 - Singleplayer"  # Name of the Minecraft window
#WINDOW_NAME="Minecraft* 1.19.4"
SHM_SOCKET="/tmp/minecraft_video"  # Shared memory socket path
WIDTH=640  # Width of the captured video
HEIGHT=360  # Height of the captured video
FRAMERATE=20  # Framerate of the video stream

# Get the window ID of the Minecraft window
WINDOW_ID=$(xwininfo -name "$WINDOW_NAME" | grep "Window id" | awk '{print $4}')

if [ -z "$WINDOW_ID" ]; then
  echo "Error: Minecraft window not found!"
  exit 1
fi

echo "Capturing Minecraft window (ID: $WINDOW_ID) and streaming to shared memory..."

# GStreamer pipeline to capture the window and stream to shared memory


gst-launch-1.0 -v\
    ximagesrc xid=$WINDOW_ID !\
    videoconvert !\
    videoscale !\
    videorate !\
    video/x-raw,format=RGB,width=$WIDTH,height=$HEIGHT,framerate=$FRAMERATE/1 !\
    queue !\
    shmsink socket-path=$SHM_SOCKET shm-size=10000000 wait-for-connection=0


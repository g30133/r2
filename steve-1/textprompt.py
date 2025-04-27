import sys
import time
import threading
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


# Ex:
# python textprompt.py
# ()> chop wood
# (chop wood)> Enter Key
# (chop wood)>
# later new prompt
# (chop wood)> climb tree
# (climb tree)>

process_running = False
process_stop_flag = threading.Event()

def process(prompt):
    global process_running
    
    process_running = True
    process_stop_flag.clear()

    for i in range(100):
        if process_stop_flag.is_set():
            break
        print(prompt)
        time.sleep(1)

    process_running = False
    print("FINISH PROCESS")
    
def main():
    print('main()')
    global process_running, process_stop_flag
    
 
    curr_input = ""

    while True:
        user_input = input(f"({curr_input})>")


        if user_input == "":
            continue

        # user gave some nonempty input, signal process to stop
        if process_running:
            process_stop_flag.set()
            process_thread.join()
            
        curr_input = user_input
        
        # start new thread
        process_thread = threading.Thread(target=process, args=(curr_input,))
        process_thread.start()

    # Main loop to keep the script running
    main_loop = GLib.MainLoop()

    # Run the main loop in a separate thread
    def run_main_loop():
        main_loop.run()

    # Start the main loop thread
    main_loop_thread = threading.Thread(target=run_main_loop)
    main_loop_thread.start()

    print('111')

    # Wait for the main loop thread to finish
    try:
        main_loop_thread.join()
    except KeyboardInterrupt:
        pass

    print('222')

if __name__ == "__main__":
    main()

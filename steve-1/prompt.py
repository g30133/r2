import mmap
import struct


HEADER_SIZE = 4 + 4 + 4
PROMPT_SIZE = 160
SHMBUF_PROMPT_SIZE = HEADER_SIZE + PROMPT_SIZE
SHMBUF_PROMPT_PATH = "/dev/shm/minecraft_prompt"
PROMPT = 'collect wood'

def init():
    #print('====init()====')
    with open(SHMBUF_PROMPT_PATH, "w+b") as f:
        f.truncate(SHMBUF_PROMPT_SIZE)
        shmbuf_prompt = mmap.mmap(f.fileno(), SHMBUF_PROMPT_SIZE, access=mmap.ACCESS_WRITE)
    #print('====init() DONE====')
    return shmbuf_prompt

    
def process(shmbuf_prompt):
    #print('====process()====')

    promptcounter = 0
    while True:
        promptcounter += 1
        
        prompt = input("Enter prompt: ")
        
        prompt = prompt.strip()
        promptlen = len(prompt)
        if promptlen == 0:
            continue
        if promptlen > PROMPT_SIZE:
            prompt = prompt[:PROMPT_SIZE]

        shmbuf_prompt.seek(HEADER_SIZE)
        shmbuf_prompt.write(prompt.encode('utf-8'))
        shmbuf_prompt.seek(4)
        shmbuf_prompt.write(struct.pack("<I", promptlen))
        shmbuf_prompt.flush()

        shmbuf_prompt.seek(0)
        shmbuf_prompt.write(struct.pack("<I", promptcounter))
        shmbuf_prompt.flush()

shmbuf_prompt = init()
process(shmbuf_prompt)

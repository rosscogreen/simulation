from PIL import Image
from gym.wrappers import Monitor

def record_videos(env, path='videos'):
    return Monitor(env, path, force=True, video_callable=lambda episode: True)

def capture_intermediate_frames(env):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame

def save_img(data, step, path='images'):
    img = Image.fromarray(data, 'RGB')
    img.save(f'{path}/step{step}.png')
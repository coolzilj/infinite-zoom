from datetime import datetime
import gradio as gr
import modules.scripts as scripts
from modules.processing import process_images
from PIL import Image
import numpy as np
import cv2


def write_video(
    file_path,
    frames,
    fps,
    reversed=True,
    start_frame_dupe_amount=15,
    last_frame_dupe_amount=30,
):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames.reverse()

    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    ## start frame duplicated
    for x in range(start_frame_dupe_amount):
        np_frame = np.array(frames[0].convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    ## last frame duplicated
    for x in range(last_frame_dupe_amount):
        np_frame = np.array(frames[len(frames) - 1].convert("RGB"))
        cv_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(cv_frame)

    writer.release()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def shrink_and_paste_on_blank(current_image, mask_width):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    """

    height = current_image.height
    width = current_image.width

    # shrink down by mask_width
    prev_image = current_image.resize((height - 2 * mask_width, width - 2 * mask_width))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)

    # create blank non-transparent image
    blank_image = np.array(current_image.convert("RGBA")) * 0
    blank_image[:, :, 3] = 1

    # paste shrinked onto blank
    blank_image[
        mask_width : height - mask_width, mask_width : width - mask_width, :
    ] = prev_image
    prev_image = Image.fromarray(blank_image)

    return prev_image


class Script(scripts.Script):
    def title(self):
        return "Infinite Zoom"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        steps = gr.Slider(
            minimum=1.0,
            maximum=100.0,
            step=1.0,
            value=5.0,
            label="Number of Outpainting Steps",
        )
        mask_width = gr.Slider(
            minimum=1.0,
            maximum=512.0,
            step=1.0,
            value=128.0,
            label="Mask Width (mask_width < image_width / 2)",
        )
        num_interpol_frames = gr.Slider(
            minimum=10.0,
            maximum=50.0,
            step=1.0,
            value=30.0,
            label="Number of Interpolation Frames Between Outpainting Steps",
        )
        fps = gr.Slider(minimum=24.0, maximum=60.0, step=1.0, value=30.0, label="FPS")
        start_frame_dupe_amount = gr.Slider(
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            value=15.0,
            label="Number of Duplicate Frames at Start",
        )
        last_frame_dupe_amount = gr.Slider(
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            value=15.0,
            label="Number of Duplicate Frames at End",
        )
        video_save_path = gr.Textbox(
            label="Video Save Path", value=str(scripts.basedir() + "/outputs")
        )
        return [
            steps,
            mask_width,
            num_interpol_frames,
            fps,
            start_frame_dupe_amount,
            last_frame_dupe_amount,
            video_save_path,
        ]

    def run(
        self,
        p,
        steps,
        mask_width,
        num_interpol_frames,
        fps,
        start_frame_dupe_amount,
        last_frame_dupe_amount,
        video_save_path,
    ):
        width = p.init_images[0].width
        height = p.init_images[0].height
        current_image = p.init_images[0]
        all_frames = []
        all_frames.append(current_image)

        for i in range(steps):
            prev_image_fix = current_image
            prev_image = shrink_and_paste_on_blank(current_image, mask_width)
            current_image = prev_image

            # create mask (black image with white mask_width width edges)
            mask_image = np.array(current_image)[:, :, 3]
            mask_image = Image.fromarray(255 - mask_image).convert("RGB")

            # inpainting step
            current_image = current_image.convert("RGB")
            p.image_mask = mask_image
            p.init_images[0] = current_image
            proc = process_images(p)
            current_image = proc.images[0].copy()
            current_image.paste(prev_image, mask=prev_image)

            # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
            for j in range(num_interpol_frames - 1):
                interpol_image = current_image
                interpol_width = round(
                    (
                        1
                        - (1 - 2 * mask_width / height)
                        ** (1 - (j + 1) / num_interpol_frames)
                    )
                    * height
                    / 2
                )
                interpol_image = interpol_image.crop(
                    (
                        interpol_width,
                        interpol_width,
                        width - interpol_width,
                        height - interpol_width,
                    )
                )

                interpol_image = interpol_image.resize((height, width))

                # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
                interpol_width2 = round(
                    (1 - (height - 2 * mask_width) / (height - 2 * interpol_width))
                    / 2
                    * height
                )
                prev_image_fix_crop = shrink_and_paste_on_blank(
                    prev_image_fix, interpol_width2
                )
                interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

                all_frames.append(interpol_image)

            all_frames.append(current_image)

        video_file_name = "infinite_zoom"
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        write_video(
            video_save_path + "/" + video_file_name + "_out_" + date_time + ".mp4",
            all_frames,
            fps,
            False,
            start_frame_dupe_amount,
            last_frame_dupe_amount,
        )
        write_video(
            video_save_path + "/" + video_file_name + "_in_" + date_time + ".mp4",
            all_frames,
            fps,
            True,
            start_frame_dupe_amount,
            last_frame_dupe_amount,
        )

        return proc

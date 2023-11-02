import os
import argparse
import cv2
import time
import json

import mediapipe as mp
model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

video_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.VIDEO)

image_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    running_mode=VisionRunningMode.IMAGE)


def detect_video_face_animation(input_path, output_path):
    print(f'Detect from {input_path} to {output_path}')
    detector = FaceLandmarker.create_from_options(video_options)
    print("Find videos")

    video_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith('.mp4')] 
    for video_file in video_files:
        print(f'Processing: {video_file}')
        input_file = os.path.join(input_path, video_file)
        output_file = os.path.join(output_path, video_file.replace('.mp4','.json'))

        cap = cv2.VideoCapture(input_file)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", frame_rate)
        start_time = time.time()
        face_blendshapes_out = {'startTime': start_time, 'endTime': start_time, 'fps': frame_rate,  'blendshapes': {}}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect_for_video(mp_image,frame_rate)
            face_blendshapes = detection_result.face_blendshapes[0]
            for face_blendshapes_category in face_blendshapes:
                if face_blendshapes_category.category_name in face_blendshapes_out['blendshapes']:
                    face_blendshapes_out['blendshapes'][face_blendshapes_category.category_name].append(face_blendshapes_category.score)
                else:
                    face_blendshapes_out['blendshapes'][face_blendshapes_category.category_name] = [face_blendshapes_category.score]
        cap.release()
        end_time = time.time()
        face_blendshapes_out['endTime'] = end_time
        with open(output_file, "w") as file:
            json.dump(face_blendshapes_out, file)

        print(f"Finshed: {input_file} -> {output_file}")

def detect_image_face(input_path, output_path):
    print(f'Detect from {input_path} to {output_path}')
    detector = FaceLandmarker.create_from_options(image_options)

    images_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith('.jpg')] 
    for img_file in images_files:
        print(f'Processing: {img_file}')
        input_file = os.path.join(input_path, img_file)
        output_file = os.path.join(output_path, img_file.replace('.jpg','.json'))

        image = mp.Image.create_from_file(input_file)
        start_time = time.time()
        face_blendshapes_out = {'startTime': start_time, 'endTime': start_time, 'fps': -1,  'blendshapes': {}}
        detection_result = detector.detect(image)
        face_blendshapes = detection_result.face_blendshapes[0]
        for face_blendshapes_category in face_blendshapes:
            face_blendshapes_out['blendshapes'][face_blendshapes_category.category_name] = [face_blendshapes_category.score]

        with open(output_file, "w") as file:
            json.dump(face_blendshapes_out, file)

        print(f"Finshed: {input_file} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video to ARKit BlendShapes')
    parser.add_argument('--input', type=str,default="./videos/", help='Input Video Path')
    parser.add_argument('--output', type=str,default="./outputs/", help='Output Path')
    parser.add_argument('--image', type=bool, default=False,  help='Detect Image')

    args = parser.parse_args()

    if args.image:
        detect_image_face(args.input, args.output)
    else:   
        detect_video_face_animation(args.input, args.output)
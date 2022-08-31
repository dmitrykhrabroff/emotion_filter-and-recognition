import streamlit as st
import torch
import cv2
import cv2 as cv
import mediapipe as mp
import tempfile
import time
import face_transformation_utils as utils
import base_utils
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from emotion_classifier_model import ResNet9
import torchvision.transforms as tt

frame_counter =0
right_eye_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX
DEMO_VIDEO = 'media\demo.mp4'
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
        39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
FONTS =cv.FONT_HERSHEY_COMPLEX
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476]
map_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

st.title('FaceEmotion Recognition and  SnapChat filter App using MediaPipe')

st.markdown(
    '''
    <style>
    [data-testid='stSidebar'][aria-expanded='true'] > div: first-child{
        width: 350px
    }
    [data-testid='stSidebar'][aria-expanded='true'] > div: first-child{
        width: 350px
        margin_left: -350px
    }
    </style>
    ''', unsafe_allow_html = True
)

st.sidebar.title('FaceMash SideBar')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = width / float(w)
        dim = (int(w*r), height)

    else:
        r = width / float(w)
        dim = (width, int(h*r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def videofilter_overlay(frame, filter_dict, mesh_coords, func, emotion, condition, green = False):
    if condition:
        filter, filter_counter = filter_dict[emotion][condition]
    else:
        filter, filter_counter = filter_dict[emotion]
    ret_filter, filter_frame = filter.read()
    filter_frame = cv2.cvtColor(filter_frame, cv2.COLOR_BGR2RGB)
    filter_counter += 1
    if filter_counter == filter.get(cv2.CAP_PROP_FRAME_COUNT):
        filter.set(cv2.CAP_PROP_POS_FRAMES, 0)
        filter_counter = 0
    if condition:
        filter_dict[emotion][condition][1] = filter_counter
    else:
        filter_dict[emotion][1] = filter_counter
    return func(frame, filter_frame, mesh_coords), filter_dict

def imagefilter_overlay(frame, filter, mesh_coords, func, emotion):
    filter_img = filter[emotion]
    return func(frame, filter_img, mesh_coords)

def check_status(ratio, face_part_status, face_count, threshold=5, delay=5):
    if ratio < threshold:
        if face_part_status[face_count][0] == 'OPEN':
            face_part_status[face_count][1] = 0
        else:
            face_part_status[face_count][1] += 1
            if face_part_status[face_count][1] > delay:
                face_part_status[face_count][0] = 'OPEN'
    else:
        if face_part_status[face_count][0] == 'CLOSED':
            face_part_status[face_count][1] = 0
        else:
            face_part_status[face_count][1] += 1
            if face_part_status[face_count][1] > delay:
                face_part_status[face_count][0] = 'CLOSED'
    return face_part_status

app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App',  'Run on Video'])

model_mode = st.sidebar.selectbox('Choose the model', ['Emotion Recognition', 'Most scary snapchat filter'])

if app_mode == 'About App':
    st.markdown('В этом приложении мы используем библиотеку MediaPipe для распознования эмоций, и с помощью нее сделаем что-то типа snapchat фильтра.')

    st.markdown(
        '''
        <style>
        [data-testid='stSidebar'][aria-expanded='true'] > div: first-child{
            width: 150px
        }
        [data-testid='stSidebar'][aria-expanded='true'] > div: first-child{
            width: 150px
            margin_left: -350px
        }
        </style>
        ''', unsafe_allow_html = True
    )
    st.markdown(
        'При подготовке данного приложения вдохновлялся следующими материалами.')
    st.subheader(
        'Видео по созданию Computer vision приложения с помощью библиотки  streamlit')
    st.video('https://www.youtube.com/watch?v=wyWmWaXapmI&list=WL&index=80&ab_channel=AugmentedStartups')
    st.subheader(
        'Overlay изображений, создание масок')
    st.video('https://www.youtube.com/watch?v=Y-mCtkv41rk&t=170s&ab_channel=AiPhile')

    st.subheader('[Статья по распознованию эмоций:](https://medium.com/swlh/emotion-detection-using-pytorch-4f6fbfd14b2e)')


    st.markdown('About Me')


elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.checkbox('Use Webcam')
    record = st.sidebar.checkbox('Record Video')

    if record:
        st.checkbox('Recording', value = True)

    max_faces = st.sidebar.number_input('Maximum number of faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    st.markdown("## Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "mov", "avi", "asf", "m4v"])

    tfile = tempfile.NamedTemporaryFile(delete = False)

    #We get our input video
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #Recording Part
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tfile.name)

    fps = 0
    i = 0
    # face bounder indices

    map_face_mesh = mp.solutions.face_mesh

    drawing_spec= mp_drawing.DrawingSpec(thickness = 1, circle_radius =1, color=(0, 0, 255))

    kpi_1, kpi_2, kpi_3  = st.beta_columns(3)
    with kpi_1:
        st.markdown('**Frame rate**')
        kpi1_text = st.markdown('0')

    with kpi_2:
        st.markdown('**Detected Faces**')
        kpi2_text = st.markdown('0')

    with kpi_3:
        st.markdown('**Image Width**')
        kpi3_text = st.markdown('0')

    st.markdown("<hr/>", unsafe_allow_html = True)


    #FaceMeshPredictor
    if model_mode == 'Emotion Recognition':
        model = ResNet9(1, 7)
        model.load_state_dict(torch.load('fer2013-resnet9.pth', map_location=torch.device('cpu')))
        classes = dict(zip(list(
            range(7)), ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]))
        res = 'neutral'
        cach_results = torch.tensor([0, 0])
        j, k = 0, 0
        frame_counter = 0
        with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            prevTime = 0
            while vid.isOpened():
                i += 1
                j += 1
                frame_counter += 1
                ret, frame = vid.read()
                face_count = 0
                if not ret:
                    continue
                if use_webcam:
                    cv2.flip(frame, 1)
                else:
                    if frame_counter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_counter = 0
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame)
                frame.flags.writeable = True

                if results.detections:
                    for detection in results.detections:
                        face_count += 1
                        location = detection.location_data
                        relative_bounding_box = location.relative_bounding_box
                        rect_start_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin, relative_bounding_box.ymin, width,
                            height)
                        rect_end_point = _normalized_to_pixel_coordinates(
                            relative_bounding_box.xmin + relative_bounding_box.width,
                            relative_bounding_box.ymin + relative_bounding_box.height, width,
                            height)
                        if not (rect_start_point and rect_end_point):
                            continue

                        xleft, ytop = rect_start_point
                        xright, ybot = rect_end_point

                        ## Lets draw a bounding box
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.rectangle(frame, rect_start_point, rect_end_point, color, thickness)
                        cv2.putText(frame, res, (xleft, ytop - 5), 0, 1.5, color, thickness,
                                    lineType=cv2.LINE_AA)
                        if j < 5:
                            j += 1
                            continue

                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        crop_img = gray[ytop: ybot, xleft: xright]
                        crop_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_AREA)
                        transform = tt.Compose([tt.ToTensor()])
                        crop_img = transform(crop_img).unsqueeze(0)
                        output = model(crop_img)
                        top_pred, top_ind = torch.topk(output, k=2, dim=1)
                        preds = top_ind[0][0].item()
                        cur_res = classes[preds]
                        j = 0
                        if preds not in cach_results and top_pred[0][0].item() / top_pred[0][1].item() > 1.3:
                            if k < 3:
                                k += 1
                            else:
                                res = cur_res
                                cach_results = top_ind
                                k = 0


                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                # DashBoard
                kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)
                kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels="BGR', use_column_width = True")


    elif model_mode == 'Most scary snapchat filter':
        with map_face_mesh.FaceMesh(max_num_faces=max_faces,refine_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            prevTime = 0
            # Initialize the VideoCapture object to read from the smoke animation video stored in the disk.
            mouth_animation = {'anger' : {'OPEN' : [cv2.VideoCapture('media/smoke_animation.mp4'), 0], 'CLOSED' : [cv2.VideoCapture('media/555.mp4'), 0]}}
            head_animation = {'anger' : [cv2.VideoCapture('media/monkey.mp4'), 0]}
            iris_filter = cv2.imread('media/vecteezy_fire_1188566.png')  # , cv2.IMREAD_UNCHANGED)
            iris_filter = cv2.cvtColor(iris_filter, cv2.COLOR_BGR2RGB)
            evil_horn = cv2.imread('media/evil_horn2.png')  # , cv2.IMREAD_UNCHANGED)
            evil_horn = cv2.cvtColor(evil_horn, cv2.COLOR_BGR2RGB)
            iris_img = {'anger' :  iris_filter}
            hat_img = {'anger' :  evil_horn,}
            mouth_status = {1: ['CLOSED',0], 2 : ['CLOSED',0]}
            left_eye_status = {1: ['CLOSED',0], 2 : ['CLOSED',0]}
            right_eye_status = {1: ['CLOSED',0], 2 : ['CLOSED',0]}

            # Set the smoke animation video frame counter to zero.
            color = (255, 0, 0)
            thickness = 2


            while vid.isOpened():
                i += 1
                ret, frame = vid.read()
                if not ret:
                    continue
                # frame_height, frame_width = frame.shape[:2]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if use_webcam:
                    frame = cv2.flip(frame, 1)

                results = face_mesh.process(frame)
                frame.flags.writeable = True
                face_count = 0
                emotion = 'anger'

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mesh_coords = base_utils.landmarksDetection(frame,  face_landmarks.landmark)

                        mouth_ratio = base_utils.isOpen(mesh_coords, 308, 15, 78, 12)
                        left_eye_ratio = base_utils.isOpen( mesh_coords, 133, 145, 33,159)
                        right_eye_ratio = base_utils.isOpen( mesh_coords, 263, 374, 362, 386)
                        face_count += 1
                        left_eye_status = check_status(left_eye_ratio, left_eye_status, face_count)
                        condition = left_eye_status[face_count][0]
                        if emotion == 'anger':
                            if condition == 'OPEN':
                                    frame = imagefilter_overlay(frame, iris_img, mesh_coords, utils.leftIrisOverlay,
                                                                    emotion)
                            right_eye_status = check_status(right_eye_ratio, right_eye_status, face_count)
                            condition = right_eye_status[face_count][0]
                            if condition == 'OPEN':
                                frame = imagefilter_overlay(frame, iris_img, mesh_coords, utils.rightIrisOverlay,
                                                             emotion)

                            mouth_status = check_status(mouth_ratio, mouth_status, face_count)
                            condition = mouth_status[face_count][0]
                            if condition == 'OPEN':
                                frame, mouth_animation = videofilter_overlay(frame, mouth_animation, mesh_coords, utils.mouth_overlay,
                                                                        emotion, condition)
                            else:
                                frame, mouth_animation = videofilter_overlay(frame, mouth_animation, mesh_coords,
                                                                             utils.noses_overlay, emotion, condition)
                            frame, head_animation = videofilter_overlay(frame, head_animation, mesh_coords, utils.head_overlay,
                                                                emotion, condition=False)
                            frame = imagefilter_overlay(frame, hat_img, mesh_coords, utils.hatOverlay,
                                                           emotion)
                            frame = utils.facepartExtractor(frame, [mesh_coords[x] for x in LEFT_EYE])
                            frame = utils.facepartExtractor(frame, [mesh_coords[x] for x in RIGHT_EYE])
                        elif emotion == 'happy':
                            mouth_status = check_status(mouth_ratio, mouth_status, face_count)
                            condition = mouth_status[face_count][0]
                            if condition == 'OPEN':
                                frame, mouth_animation = videofilter_overlay(frame, mouth_animation, mesh_coords,
                                                                             utils.mouth_overlay,
                                                                             emotion, condition, green=True)
                            else:
                                frame = utils.facepartExtractor(frame, [mesh_coords[x] for x in LIPS], scale_x =1.5, scale_y=1.5)
                            frame = imagefilter_overlay(frame, hat_img, mesh_coords, utils.hatOverlay,
                                                        emotion)

                            # frame = utils.facepartExtractor(frame, [mesh_coords[x] for x in RIGHT_EYE])






                #FPS counter
                currTime =time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                #DashBoard
                kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html = True)
                kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)
                kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{mouth_status}</h1>", unsafe_allow_html = True)

                st.sidebar.text(f'frame.shape:{frame.shape}')

                frame = cv2.resize(frame, (0,0), fx = 0.8, fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                stframe.image(frame, channels = "BGR', use_column_width = True")

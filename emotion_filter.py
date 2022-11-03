import streamlit as st
import torch
import cv2
import mediapipe as mp
import tempfile
import time
import base_utils
from emotion_classifier_model import ResNet9
import torchvision.transforms as tt
from oop_model import  *


frame_counter =0
right_eye_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX
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
FONTS =cv2.FONT_HERSHEY_COMPLEX
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

def cropFaces(frame, coords):
    img_height, img_width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    max_x, min_x = (max(coords, key=lambda item: item[0]))[0], \
                   (min(coords, key=lambda item: item[0]))[0]
    max_y, min_y = (max(coords, key=lambda item: item[1]))[1], \
                   (min(coords, key=lambda item: item[1]))[1]
    cropped_img = gray[min_y: max_y, min_x: max_x]
    return cropped_img


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
        models = dict(zip(['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"],
                           [AngryFilter, DisgustFilter, FearFilter, HappyFilter, NeutralFilter, SadnessFilter, SurpriseFilter] ))
        res = 'Neutral'
        # tracemalloc.start()
        cach_results = torch.tensor([0, 0])
        j, k = 0, 0
        frame_counter = 0
        with map_face_mesh.FaceMesh(max_num_faces=max_faces, refine_landmarks=True,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            prevTime = 0
            while vid.isOpened():
                i += 1
                frame_counter += 1
                ret, frame = vid.read()
                frame = image_resize(image=frame, width=640)
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
                results = face_mesh.process(frame)
                frame.flags.writeable = True

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1
                        mesh_coords = base_utils.landmarksDetection(frame, face_landmarks.landmark)
                        cv2.putText(frame, res, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255, 0, 0), thickness = 2)
                        if j == 2:
                            gray = cropFaces(frame, mesh_coords)
                            crop_img = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
                            transform = tt.Compose([tt.ToTensor()])
                            crop_img = transform(crop_img).unsqueeze(0)
                            output = model(crop_img)
                            top_pred, top_ind = torch.topk(output, k=2, dim=1)
                            preds = top_ind[0][0].item()
                            cur_res = classes[preds]
                            j = 0
                            if preds not in cach_results and top_pred[0][0].item() / top_pred[0][1].item() > 1.3:
                                if k < 7:
                                    k += 1
                                else:
                                    res = cur_res
                                    cach_results = top_ind
                                    k = 0
                        else:
                            j += 1
                        obj = models[res](face_count)
                        # print(obj.__dict__['filters']['mouth'][1], 'obj')
                        frame = obj.forward(frame, mesh_coords)
                        frame = cv2.GaussianBlur(frame, (3,3), 0)
                        # snapshot = tracemalloc.take_snapshot()
                        # display_top(snapshot)



                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                if record:
                    out.write(frame)

                # DashBoard
                kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html = True)
                kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)

                # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                # frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels="BGR', use_column_width = True")



# Colors
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (200, 200, 200)

# Camera Settings
CAMERA_SRC = 0
FRAME_WIDTH = 640
PROCESS_WIDTH = 320

# Recognition Settings
RECOG_DIST_THRESH = 0.55
STREAK_THRESHOLD = 4

# Liveness Detection Settings
EAR_THRESHOLD = 0.22  # Eye Aspect Ratio Threshold
EAR_CONSEC_FRAMES = 2  # Consecutive frames for blink detection
REQUIRED_BLINKS = 2  # Required number of blinks
LIVENESS_TIMEOUT_FRAMES = 300  # Liveness timeout in frames (5 seconds at 30 FPS)

# Head Movement Settings
HEAD_MOVEMENT_FRAMES = 10
MIN_CENTROID_MOVEMENT = 0.7
MAX_CENTROID_MOVEMENT = 6.0
CENTROID_LANDMARK_INDICES = [30, 33, 8, 36, 45, 48, 54]

# Head Pose Settings
POSE_HISTORY_FRAMES = 15
MIN_POSE_STD_DEV_SUM = 0.45
POSE_LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]
LOOK_LEFT_RIGHT_ANGLE_THRESH = 20.0  # Minimum Yaw angle deviation for left/right look (degrees)
POSE_CENTER_THRESHOLD = 10
 # More lenient ranges for each angle
PITCH_RANGE = (-45, 45)   # Looking up/down
YAW_RANGE = (-90, 90)     # Looking left/right
ROLL_RANGE = (-45, 45)    # Head tilt

# Door/Servo Settings
SERVO_PIN = 18
SERVO_MIN_PULSE = 0.0005
SERVO_MAX_PULSE = 0.0025
SERVO_FRAME_WIDTH = 0.02
CLOSED_ANGLE = 10
OPEN_ANGLE = -70
DOOR_OPEN_TIME = 5.0

# File Paths
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_REC_MODEL_PATH = 'dlib_face_recognition_resnet_model_v1.dat'
KNOWN_FACES_DB_PATH = 'known_faces.json'

# UI Layout
UI_BUTTON_HEIGHT = 40
UI_BUTTON_WIDTH = 120
UI_BUTTON_MARGIN = 10
UI_INFO_POS = (10, 20)
UI_STATUS_POS = (10, 45)
UI_LIVENESS_POS = (10, 70)
UI_INPUT_POS = (10, 360)
UI_BUTTON_START_Y = 400

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7816607327:AAHX8qC5JZ7aQqdGDHOxiyiiS4WXkGhkokQ"
TELEGRAM_CHAT_ID = "1528417477"
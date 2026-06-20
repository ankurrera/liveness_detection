import cv2
import numpy as np
import time
import threading
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from app.database import SessionLocal
from app.services.db_services import (
    start_new_state_session, 
    close_active_session, 
    resolve_orphaned_sessions, 
    get_or_create_default_employee
)
from app.config import settings

def draw_rounded_rect(img, pt1, pt2, color, thickness, r):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w = x2 - x1
    h = y2 - y1
    r = int(min(r, w // 2, h // 2))
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)

def draw_filled_rounded_rect(img, pt1, pt2, color, r):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    w = x2 - x1
    h = y2 - y1
    r = int(min(r, w // 2, h // 2))
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1, lineType=cv2.LINE_AA)

class CVMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.latest_frame = None  # where we stash the jpeg bytes for the stream
        
        # state machine variables (make sure these are thread-safe!)
        self.current_status = "ABSENT"
        self.confidence = 0.0
        self.last_present_time = time.time()
        self.last_movement_time = time.time()
        self.state_entered_time = time.time()
        
        # keeping tabs on our tracking scores
        self.latest_raw_score = 0.0
        self.latest_smoothed_score = 0.0
        self.frame_scores = deque(maxlen=150)  # rolling window of about 10 seconds (assuming 15fps)
        
        self.lock = threading.Lock()
        self.is_mock = False
        self.prev_landmarks = {}
        
        # who are we watching right now?
        self.active_employee_id = settings.DEFAULT_EMPLOYEE_ID
        
        # blink tracking stuff
        self.blink_count = 0
        self.prev_blinking = False
        
        # yolo tracking and jitter reduction
        self.smoothed_landmarks = {}
        self.smoothed_kp_xy = None
        self.locked_track_id = None
        self.smoothed_bbox = None
        self.active_employee_name = "Ankur Bag"
        self._update_employee_name_cache()

    def _update_employee_name_cache(self):
        db = SessionLocal()
        try:
            from app.models import Employee
            emp = db.query(Employee).filter(Employee.employee_id == self.active_employee_id).first()
            if emp:
                self.active_employee_name = emp.name
            else:
                self.active_employee_name = f"Subject #{self.active_employee_id}"
        except Exception as e:
            print(f"[CV Engine] Cache Name Error: {e}")
            self.active_employee_name = f"Subject #{self.active_employee_id}"
        finally:
            db.close()



    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_current_state(self):
        with self.lock:
            return {
                "status": self.current_status,
                "confidence": self.confidence,
                "time_in_state": int(time.time() - self.state_entered_time),
                "is_mock": self.is_mock
            }

    def get_debug_metrics(self) -> dict:
        """Spits out the live numbers for the dev dashboard."""
        with self.lock:
            now = time.time()
            idle_cd = 0
            
            if self.current_status == "WORKING":
                elapsed_low = now - self.last_movement_time
                # ticking down from 60 seconds if they stop moving
                idle_cd = max(0, int(settings.IDLE_TIMEOUT - elapsed_low))
                
            return {
                "raw_score": round(self.latest_raw_score, 3),
                "smoothed_score": round(self.latest_smoothed_score, 3),
                "movement_threshold": settings.YOLO_ACTIVITY_THRESHOLD,  # the line between working and slacking
                "idle_countdown": idle_cd,
                "working_countdown": 0,
                "epsilon_filter": settings.YOLO_EPSILON_FILTER
            }

    def set_active_employee(self, employee_id: int):
        """Safely switch who we are watching without breaking the database sessions."""
        with self.lock:
            if self.active_employee_id == employee_id:
                return
            
            print(f"[CV Engine] Switching active monitored subject from ID {self.active_employee_id} to ID {employee_id}")
            
            db = SessionLocal()
            try:
                # 1. wrap up whatever the previous person was doing
                close_active_session(db, self.active_employee_id)
                
                # 2. lock onto the new person
                self.active_employee_id = employee_id
                self._update_employee_name_cache()
                
                # 3. start tracking their activity right away
                start_new_state_session(
                    db,
                    self.active_employee_id,
                    self.current_status,
                    self.confidence,
                    raw_score=self.latest_raw_score,
                    smoothed_score=self.latest_smoothed_score,
                    transition_reason=f"Switched monitoring subject to employee ID {employee_id}.",
                    notes=f"Switched monitoring subject to employee ID {employee_id}."
                )
            except Exception as e:
                print(f"[CV Engine] Error switching active employee: {e}")
            finally:
                db.close()
                
            # wipe the slate clean for the new person
            self.frame_scores.clear()
            self.prev_landmarks = {}
            self.blink_count = 0
            self.prev_blinking = False

    def _run_loop(self):
        # 1. clean up the database and set up our default users
        db = SessionLocal()
        try:
            resolve_orphaned_sessions(db)
            employee = get_or_create_default_employee(db)
            # pick the person we are supposed to be watching
            employee_id = self.active_employee_id
            self._update_employee_name_cache()
            
            # assume they aren't there until we see them
            start_new_state_session(
                db, 
                employee_id, 
                self.current_status, 
                self.confidence, 
                raw_score=0.0,
                smoothed_score=0.0,
                transition_reason="Initial state on monitoring start.",
                notes="Initial state on monitoring start."
            )
        except Exception as e:
            print(f"[CV Engine] Error during startup database prep: {e}")
        finally:
            db.close()

        # 2. try to fire up the webcam
        cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        
        # lock the resolution so we can process frames quickly
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened() or settings.MOCK_CAMERA:
            print(f"[CV Engine] Webcam index {settings.CAMERA_INDEX} not available or mock mode forced. Starting simulated feed...")
            self.is_mock = True
        else:
            print(f"[CV Engine] Successfully opened webcam on index {settings.CAMERA_INDEX}")
            self.is_mock = False

        # 3. load up the big brain yolo model
        try:
            print(f"[CV Engine] Initializing YOLOv8 Pose model ({settings.YOLO_MODEL_NAME}) on device '{settings.YOLO_DEVICE}'...")
            yolo_model = YOLO(settings.YOLO_MODEL_NAME)
            print(f"[CV Engine] YOLOv8 Pose model {settings.YOLO_MODEL_NAME} loaded successfully.")
        except Exception as e:
            print(f"[CV Engine] Failed to load YOLOv8 model: {e}. Falling back to Mock Mode.")
            self.is_mock = True
            yolo_model = None

        # 3.5 load up google's face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        prev_time = time.time()

        # the main event loop
        try:
            while self.running:
                loop_start = time.time()
                frame = None
                detected = False
                model_confidence = 0.0
                current_landmarks = {}
                is_blinking = False
                ear = 0.3

                if self.is_mock:
                    # we're in mock mode, so generate a fake video frame
                    frame, detected, model_confidence, current_landmarks = self._generate_mock_frame()
                    
                    # make the fake person blink so it looks somewhat real
                    cycle_time = int(time.time()) % 90
                    if cycle_time < 70:  # someone is here
                         sec_of_cycle = time.time() % 4
                         is_blinking = (sec_of_cycle < 0.25)  # give a quick blink every 4 seconds
                    
                    time.sleep(max(0.01, (1.0 / 15.0) - (time.time() - loop_start)))  # throttle the fake video to ~15fps
                else:
                    # real webcam mode - grab a frame from the hardware
                    ret, frame = cap.read()
                    if not ret:
                        print("[CV Engine] Failed to grab frame. Falling back to Mock Mode...")
                        self.is_mock = True
                        continue

                    # flip the image like a mirror so it doesn't mess with your head
                    frame = cv2.flip(frame, 1)
                    h, w, c = frame.shape

                    # both yolo and mediapipe expect rgb, not opencv's default bgr
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # let mediapipe find the face landmarks
                    face_results = face_mesh.process(rgb_frame)
                    
                    # let yolo find the body joints
                    if yolo_model is None:
                        self.is_mock = True
                        continue
                    
                    # run the tracker to keep ids consistent across frames
                    yolo_results = yolo_model.track(
                        rgb_frame,
                        persist=True,
                        tracker=settings.YOLO_TRACKER,
                        imgsz=settings.YOLO_INFERENCE_SIZE,
                        conf=settings.YOLO_CONFIDENCE_THRESHOLD,
                        device=settings.YOLO_DEVICE,
                        verbose=False
                    )
                    
                    pose_detected = False
                    face_detected = False
                    detected = False
                    model_confidence = 0.0
                    if face_results.multi_face_landmarks:
                        landmarks = face_results.multi_face_landmarks[0].landmark
                        def get_pt(idx):
                            return np.array([landmarks[idx].x, landmarks[idx].y])
                        
                        # figure out how wide the left eye is open
                        le_v1 = np.linalg.norm(get_pt(160) - get_pt(144))
                        le_v2 = np.linalg.norm(get_pt(158) - get_pt(153))
                        le_h = np.linalg.norm(get_pt(33) - get_pt(133))
                        ear_left = (le_v1 + le_v2) / (2.0 * le_h)
                        
                        # figure out how wide the right eye is open
                        re_v1 = np.linalg.norm(get_pt(385) - get_pt(380))
                        re_v2 = np.linalg.norm(get_pt(387) - get_pt(373))
                        re_h = np.linalg.norm(get_pt(362) - get_pt(263))
                        ear_right = (re_v1 + re_v2) / (2.0 * re_h)
                        
                        ear = (ear_left + ear_right) / 2.0
                        if ear < 0.22:
                            is_blinking = True

                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        boxes = yolo_results[0].boxes
                        
                        # figure out which person we should actually be tracking if there are multiple
                        # find the biggest bounding box
                        xyxy_array = boxes.xyxy.cpu().numpy()
                        areas = (xyxy_array[:, 2] - xyxy_array[:, 0]) * (xyxy_array[:, 3] - xyxy_array[:, 1])
                        
                        # try to grab the track ids if bytetrack assigned them
                        track_ids = None
                        if boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy().astype(int)
                        
                        # stick with the person we were already watching, or pick the biggest person
                        primary_idx = 0
                        if track_ids is not None and self.locked_track_id is not None and self.locked_track_id in track_ids:
                            primary_idx = int(np.where(track_ids == self.locked_track_id)[0][0])
                        else:
                            primary_idx = int(np.argmax(areas))
                            if track_ids is not None:
                                self.locked_track_id = int(track_ids[primary_idx])
                        
                        kp_xyn = yolo_results[0].keypoints.xyn[primary_idx].cpu().numpy()  # shape (17, 2)
                        kp_xy = yolo_results[0].keypoints.xy[primary_idx].cpu().numpy()    # shape (17, 2)
                        kp_conf = yolo_results[0].keypoints.conf[primary_idx].cpu().numpy()  # shape (17,)
                        
                        # only judge confidence based on what we can actually see, don't penalize for hidden legs
                        visible_confs = kp_conf[kp_conf > 0.2]
                        model_confidence = float(np.mean(visible_confs)) if len(visible_confs) > 0 else float(np.mean(kp_conf))
                        
                        if model_confidence > 0.3:
                            pose_detected = True
                            face_detected = True
                            detected = True

                        # smooth out the keypoints so they don't jump around like crazy
                        alpha = settings.YOLO_SMOOTHING_FACTOR
                        if self.smoothed_kp_xy is None or len(self.smoothed_kp_xy) != len(kp_xy):
                            self.smoothed_kp_xy = kp_xy.copy()
                        else:
                            for idx in range(len(kp_xy)):
                                if kp_conf[idx] > 0.3:
                                    self.smoothed_kp_xy[idx] = alpha * kp_xy[idx] + (1 - alpha) * self.smoothed_kp_xy[idx]
                                else:
                                    self.smoothed_kp_xy[idx] = kp_xy[idx]

                        # map the yolo joints to the mediapipe-style index
                        # yolo index: 0=nose, 5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow, 9=l_wrist, 10=r_wrist
                        # our internal index: nose=0, shoulders=11/12, elbows=13/14, wrists=15/16
                        for yolo_idx, mp_idx in [(0, 0), (5, 11), (6, 12), (7, 13), (8, 14), (9, 15), (10, 16)]:
                            if kp_conf[yolo_idx] > 0.3:
                                current_landmarks[mp_idx] = (float(kp_xyn[yolo_idx][0]), float(kp_xyn[yolo_idx][1]))
                        
                        # apply the same smoothing to our movement analysis coordinates
                        for mp_idx, val in current_landmarks.items():
                            if mp_idx in self.smoothed_landmarks:
                                prev_val = self.smoothed_landmarks[mp_idx]
                                self.smoothed_landmarks[mp_idx] = (
                                    alpha * val[0] + (1 - alpha) * prev_val[0],
                                    alpha * val[1] + (1 - alpha) * prev_val[1]
                                )
                            else:
                                self.smoothed_landmarks[mp_idx] = val
                        
                        # skipping the body skeleton for now to keep the UI clean

                        # paint the cool cybernetic face mesh on top
                        self._draw_face_mesh(frame, face_results)

                # count the blinks and use them to prove they are still alive/working
                if is_blinking:
                    with self.lock:
                        if not self.prev_blinking:
                            self.blink_count += 1
                            self.last_movement_time = time.time()  # blinks count as movement! keep them working!
                            print(f"[CV Engine] Blink detected (Count: {self.blink_count})")
                        self.prev_blinking = True
                else:
                    with self.lock:
                        self.prev_blinking = False

                # 4. calculate how much they actually moved, filtering out tiny jitters
                raw_score = 0.0
                epsilon = settings.YOLO_EPSILON_FILTER
                
                # give more weight to hand movements than just head bobbing
                groups = {
                    "hands": {"indices": [15, 16], "weight": 0.50},      # hands matter the most
                    "arms": {"indices": [13, 14], "weight": 0.25},       # elbows are secondary
                    "shoulders": {"indices": [11, 12], "weight": 0.15},  # shoulders a bit less
                    "head": {"indices": [0], "weight": 0.10}            # head movement is the least important indicator of work
                }
                
                group_displacements = {}
                if detected and self.smoothed_landmarks and self.prev_landmarks:
                    for group_name, info in groups.items():
                        sum_d = 0.0
                        count = 0
                        for idx in info["indices"]:
                            if idx in self.smoothed_landmarks and idx in self.prev_landmarks:
                                pt = self.smoothed_landmarks[idx]
                                prev_pt = self.prev_landmarks[idx]
                                d = np.sqrt((pt[0] - prev_pt[0])**2 + (pt[1] - prev_pt[1])**2)
                                
                                # ignore movements that are too small (probably just noise)
                                if d >= epsilon:
                                    sum_d += d
                                count += 1
                        if count > 0:
                            group_displacements[group_name] = sum_d / count

                # adjust the math if we can't see all their joints (like hands under the desk)
                if group_displacements:
                    total_available_weight = sum(groups[g]["weight"] for g in group_displacements.keys())
                    if total_available_weight > 0:
                        weighted_sum = 0.0
                        for g, disp in group_displacements.items():
                            normalized_w = groups[g]["weight"] / total_available_weight
                            weighted_sum += normalized_w * disp
                        # make the score a nice readable number (0.0 to 1.0+)
                        raw_score = weighted_sum * 100.0

                # give them a massive activity boost if they blink, so they don't get marked idle while just staring at code
                if detected and is_blinking:
                    raw_score += 25.0
                
                # save these coordinates so we can compare them next time
                if detected:
                    self.prev_landmarks = self.smoothed_landmarks.copy()
                else:
                    self.prev_landmarks = {}

                # 5. average out the scores over time to prevent flickering
                self.frame_scores.append(raw_score)
                smoothed_score = sum(self.frame_scores) / len(self.frame_scores) if self.frame_scores else 0.0
                
                with self.lock:
                    self.latest_raw_score = raw_score
                    self.latest_smoothed_score = smoothed_score

                # 6. the brain of the operation - deciding if they are working, idle, or gone
                now = time.time()
                new_status = self.current_status
                old_status = self.current_status
                state_changed = False
                transition_reason = ""

                with self.lock:
                    if detected:
                        self.last_present_time = now
                        self.confidence = float(model_confidence)
                        is_active_working = smoothed_score >= settings.YOLO_ACTIVITY_THRESHOLD  # did they hit the threshold?

                        if self.current_status == "ABSENT":
                            # they just got back, immediately mark them as working
                            new_status = "WORKING"
                            self.last_movement_time = now
                            transition_reason = "Subject detected returned in frame."
                        elif is_active_working:
                            # they moved enough to reset the idle timer
                            self.last_movement_time = now
                            if self.current_status == "IDLE":
                                new_status = "WORKING"
                                transition_reason = f"Active movement resumed (score: {smoothed_score:.3f})."
                        else:
                            # they aren't moving enough
                            if self.current_status == "WORKING":
                                # have they been slacking for a full 60 seconds?
                                elapsed_low = now - self.last_movement_time
                                if elapsed_low >= settings.IDLE_TIMEOUT:
                                    new_status = "IDLE"
                                    transition_reason = f"Low activity persisted for 60 seconds (average score: {smoothed_score:.3f})."
                    else:
                        # we can't see anyone
                        self.confidence = 0.0
                        if self.current_status != "ABSENT":
                            # are they really gone, or just dropped something on the floor? (wait 10s)
                            if now - self.last_present_time >= settings.ABSENT_TIMEOUT:
                                new_status = "ABSENT"
                                self.locked_track_id = None
                                self.smoothed_kp_xy = None
                                self.smoothed_landmarks = {}
                                transition_reason = "No person detected for 10 continuous seconds."

                    # make it official
                    if new_status != self.current_status:
                        print(f"[CV Engine] Transition: {self.current_status} -> {new_status} | Reason: {transition_reason}")
                        self.current_status = new_status
                        self.state_entered_time = now
                        state_changed = True

                # 7. write the change to the database
                if state_changed:
                    db = SessionLocal()
                    try:
                        start_new_state_session(
                            db, 
                            self.active_employee_id, 
                            self.current_status, 
                            self.confidence,
                            raw_score=self.latest_raw_score,
                            smoothed_score=self.latest_smoothed_score,
                            transition_reason=transition_reason,
                            notes=transition_reason
                        )
                    except Exception as e:
                        print(f"[CV Engine] DB Session Log Error: {e}")
                    finally:
                        db.close()

                # 8. calculate fps and draw the dev dashboard
                current_fps = 1.0 / (time.time() - prev_time + 1e-6)
                prev_time = time.time()

                # slap the hud onto the video frame
                self._draw_hud(frame, current_fps, raw_score, smoothed_score)

                # 9. squish it down into a jpeg so we can stream it
                ret_enc, jpeg_buffer = cv2.imencode('.jpg', frame)
                if ret_enc:
                    self.latest_frame = jpeg_buffer.tobytes()

        finally:
            if not self.is_mock:
                cap.release()
            
            face_mesh.close()
            
            db = SessionLocal()
            try:
                close_active_session(db, self.active_employee_id)
                print("[CV Engine] Gracefully closed active session segment on shutdown.")
            except Exception as e:
                print(f"[CV Engine] DB shutdown segment close failed: {e}")
            finally:
                db.close()
            print("[CV Engine] Thread stopped successfully.")

    def _draw_hud(self, frame, fps, raw_score, smoothed_score):
        h, w, c = frame.shape
        # background box for the dashboard so text is readable
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 140), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # colors for different states
        status_colors = {
            "WORKING": (74, 163, 22),  # #16A34A (Green)
            "IDLE": (11, 158, 245),    # #F59E0B (Amber)
            "ABSENT": (68, 68, 239)    # #EF4444 (Red)
        }
        color = status_colors.get(self.current_status, (255, 255, 255))

        # draw the dev metrics on the screen
        cv2.putText(frame, f"STATE: {self.current_status}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not self.is_mock and self.locked_track_id is not None:
            cv2.putText(frame, f"Track ID: {self.locked_track_id}", (180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Raw Score: {raw_score:.3f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Smooth Score: {smoothed_score:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.textWidth = 10
        
        # show the countdown timers if applicable
        now = time.time()
        if self.current_status == "WORKING":
            elapsed = now - self.last_movement_time
            cd = max(0, int(settings.IDLE_TIMEOUT - elapsed))
            cv2.putText(frame, f"Idle Countdown: {cd}s", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        else:
            cv2.putText(frame, f"Idle Countdown: --", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        mode_str = "SIMULATED FEED" if self.is_mock else "WEBCAM LIVE"
        cv2.putText(frame, f"Mode: {mode_str} ({fps:.1f} FPS)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # draw the little colored circle in the top right
        cv2.circle(frame, (w - 25, 25), 8, color, -1)
        cv2.circle(frame, (w - 25, 25), 10, (255, 255, 255), 1)

    def _draw_hand_skeleton(self, frame, w_pt, e_pt, w_conf, e_conf):
        if w_conf > 0.3 and e_conf > 0.3:
            dx = w_pt[0] - e_pt[0]
            dy = w_pt[1] - e_pt[1]
            dist = np.sqrt(dx**2 + dy**2) + 1e-6
            
            ux = dx / dist
            uy = dy / dist
            
            hand_len = max(15, min(30, int(dist * 0.35)))
            
            angles = [-0.6, -0.25, 0.0, 0.25, 0.5]
            finger_scales = [0.75, 0.95, 1.0, 0.95, 0.8]
            knuckles = []
            
            for i, a in enumerate(angles):
                cos_a = np.cos(a)
                sin_a = np.sin(a)
                rx = ux * cos_a - uy * sin_a
                ry = ux * sin_a + uy * cos_a
                
                k_len = hand_len * 0.45
                k_pt = (int(w_pt[0] + rx * k_len), int(w_pt[1] + ry * k_len))
                knuckles.append(k_pt)
                
                f_len = hand_len * finger_scales[i]
                jt = (int(k_pt[0] + rx * f_len * 0.4), int(k_pt[1] + ry * f_len * 0.4))
                tip = (int(k_pt[0] + rx * f_len * 0.8), int(k_pt[1] + ry * f_len * 0.8))
                
                # connect the wrist to the knuckles
                cv2.line(frame, (int(w_pt[0]), int(w_pt[1])), k_pt, (235, 235, 235), 1)
                # knuckle to joint
                cv2.line(frame, k_pt, jt, (235, 235, 235), 1)
                # joint to fingertip
                cv2.line(frame, jt, tip, (235, 235, 235), 1)
                
                # draw the little green joint dots
                cv2.circle(frame, k_pt, 1, (74, 163, 22), -1)
                cv2.circle(frame, jt, 1, (74, 163, 22), -1)
                cv2.circle(frame, tip, 1, (74, 163, 22), -1)
                
            # connect the knuckles across the palm
            for i in range(len(knuckles) - 1):
                cv2.line(frame, knuckles[i], knuckles[i+1], (235, 235, 235), 1)

    def _draw_face_mesh(self, frame, face_results):
        if not face_results or not face_results.multi_face_landmarks:
            return
            
        mesh_color = (153, 211, 52)
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=mesh_color)
        
        for face_landmarks in face_results.multi_face_landmarks:
            if settings.YOLO_DEBUG_MODE:
                # draw the super intense dense mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
            else:
                # just draw the clean contours and eyes for production
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)


    def _generate_mock_frame(self):
        """
        Spits out fake frames of someone sitting at a desk.
        This is just for testing if you don't have a webcam.
        It loops through working -> idle -> absent.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (24, 20, 15)  # dark slate background

        # draw a blocky table
        cv2.rectangle(frame, (40, 390), (600, 480), (35, 42, 50), -1)
        cv2.rectangle(frame, (210, 280), (430, 390), (60, 60, 60), -1)
        cv2.rectangle(frame, (220, 290), (420, 375), (10, 10, 10), -1)
        cv2.rectangle(frame, (300, 390), (340, 420), (50, 50, 50), -1)

        cycle_time = int(time.time()) % 90
        
        detected = False
        confidence = 0.0
        landmarks = {}

        if cycle_time < 70:
            # they are at the desk
            detected = True
            t = time.time()
            
            # configure how much they move based on the simulated state
            if cycle_time < 35:
                # they are working, so simulate lots of jitter and typing
                confidence = 0.94
                noise_x = int(np.sin(t * 8) * 4)
                noise_y = int(np.cos(t * 10) * 3)
                noise_hand_l = int(np.sin(t * 12) * 8)
                noise_hand_r = int(np.cos(t * 15) * 6)
                
                head_center = (320 + noise_x, 170 + noise_y)
                l_shoulder = (260, 250 + noise_y // 2)
                r_shoulder = (380, 250 + noise_y // 2)
                l_elbow = (240, 310 + noise_x)
                r_elbow = (400, 310 + noise_y)
                l_wrist = (270 + noise_hand_l, 350 + noise_hand_l // 2)
                r_wrist = (370 + noise_hand_r, 350 + noise_hand_r // 2)
                
                cv2.rectangle(frame, (240, 400), (400, 420), (74, 163, 22), 1)
                cv2.putText(frame, "SIMULATING ACTIVE WORK", (250, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (74, 163, 22), 1)
            else:
                # they are slacking, just barely moving
                confidence = 0.90
                drift_x = int(np.sin(t * 0.4) * 0.4)
                drift_y = int(np.cos(t * 0.4) * 0.4)
                
                head_center = (330 + drift_x, 180 + drift_y)
                l_shoulder = (270, 260)
                r_shoulder = (390, 260)
                l_elbow = (240, 320)
                r_elbow = (420, 320)
                l_wrist = (230, 350)
                r_wrist = (410, 350)
                
                cv2.putText(frame, "SIMULATING STATIC IDLE", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 158, 245), 1)

            # 1. draw the dark silhouette of a person
            cv2.circle(frame, head_center, 28, (45, 40, 35), -1)
            cv2.line(frame, l_shoulder, r_shoulder, (45, 40, 35), 16)
            cv2.line(frame, l_shoulder, l_elbow, (45, 40, 35), 8)
            cv2.line(frame, l_elbow, l_wrist, (45, 40, 35), 8)
            cv2.line(frame, r_shoulder, r_elbow, (45, 40, 35), 8)
            cv2.line(frame, r_elbow, r_wrist, (45, 40, 35), 8)

            # 2. draw the cool cyber skeleton on top
            # thin white lines for arms
            cv2.line(frame, l_shoulder, l_elbow, (235, 235, 235), 1)
            cv2.line(frame, l_elbow, l_wrist, (235, 235, 235), 1)
            cv2.line(frame, r_shoulder, r_elbow, (235, 235, 235), 1)
            cv2.line(frame, r_elbow, r_wrist, (235, 235, 235), 1)
            
            # chest and torso lines
            s_mid_x = int((l_shoulder[0] + r_shoulder[0]) / 2.0)
            s_mid_y = int((l_shoulder[1] + r_shoulder[1]) / 2.0)
            # neck
            cv2.line(frame, head_center, (s_mid_x, s_mid_y), (235, 235, 235), 1)
            
            # hips
            h_mid_x = 320
            h_mid_y = 370
            # spine
            cv2.line(frame, (s_mid_x, s_mid_y), (h_mid_x, h_mid_y), (235, 235, 235), 1)
            # hip bones
            cv2.line(frame, (280, h_mid_y), (360, h_mid_y), (235, 235, 235), 1)
            cv2.line(frame, l_shoulder, (280, h_mid_y), (235, 235, 235), 1)
            cv2.line(frame, r_shoulder, (360, h_mid_y), (235, 235, 235), 1)
            
            # x-brace across the chest
            cv2.line(frame, l_shoulder, (360, h_mid_y), (220, 220, 220), 1)
            cv2.line(frame, r_shoulder, (280, h_mid_y), (220, 220, 220), 1)

            # the green dots on the joints
            cv2.circle(frame, l_shoulder, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_shoulder, 2, (74, 163, 22), -1)
            cv2.circle(frame, l_elbow, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_elbow, 2, (74, 163, 22), -1)
            cv2.circle(frame, l_wrist, 2, (74, 163, 22), -1)
            cv2.circle(frame, r_wrist, 2, (74, 163, 22), -1)

            # draw the fake hands
            self._draw_hand_skeleton(frame, l_wrist, l_elbow, 1.0, 1.0)
            self._draw_hand_skeleton(frame, r_wrist, r_elbow, 1.0, 1.0)

            # 3. Draw thin cybernetic face mesh contours using mock coordinates
            mock_nose = (int(head_center[0]), int(head_center[1]))
            mock_eye_l = (int(head_center[0] - 8), int(head_center[1] - 2))
            mock_eye_r = (int(head_center[0] + 8), int(head_center[1] - 2))
            
            is_blinking_mock = (t % 4 < 0.25) # Blink for 0.25s every 4s
            cv2.circle(frame, mock_nose, 2, (153, 211, 52), -1)
            
            if is_blinking_mock:
                # Closed eyes: draw flat indicator eye lines
                cv2.line(frame, (mock_eye_l[0]-4, mock_eye_l[1]), (mock_eye_l[0]+4, mock_eye_l[1]), (20, 220, 240), 1)
                cv2.line(frame, (mock_eye_r[0]-4, mock_eye_r[1]), (mock_eye_r[0]+4, mock_eye_r[1]), (20, 220, 240), 1)
                cv2.putText(frame, "BLINK", (mock_nose[0]-16, mock_nose[1]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 220, 240), 1)
            else:
                cv2.circle(frame, mock_eye_l, 1, (153, 211, 52), -1)
                cv2.circle(frame, mock_eye_r, 1, (153, 211, 52), -1)

            # Export landmarks for main loop displacement scoring
            landmarks[0] = (head_center[0]/640.0, head_center[1]/480.0)
            landmarks[11] = (l_shoulder[0]/640.0, l_shoulder[1]/480.0)
            landmarks[12] = (r_shoulder[0]/640.0, r_shoulder[1]/480.0)
            landmarks[15] = (l_wrist[0]/640.0, l_wrist[1]/480.0)
            landmarks[16] = (r_wrist[0]/640.0, r_wrist[1]/480.0)
            
        else:
            # ABSENT state
            detected = False
            confidence = 0.0
            landmarks = {}
            cv2.putText(frame, "EMPTY CHAIR (ABSENT)", (240, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (68, 68, 239), 1)

        return frame, detected, confidence, landmarks

# Create a single global monitor instance
cv_monitor = CVMonitor()

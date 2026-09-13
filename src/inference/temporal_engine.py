from collections import deque
import time


class DriverTemporalEngine:
    """
    Real-time temporal reasoning for the Driver Monitoring System.

    This layer consumes frame-level model predictions and turns them into
    stable safety states. It deliberately does not perform ML itself.

    Design goals for live webcam use:
      - reason from capture timestamps, not processing speed
      - tolerate short detector dropouts without resetting a violation
      - never let stale PERCLOS alone force DROWSY after eyes reopen
      - preserve bilateral-eye evidence for drowsiness
      - make yawning/phone/seatbelt persistence robust to missed frames
    """

    def __init__(
        self,
        eye_closure_threshold=1.5,
        warning_closure_threshold=0.9,
        blink_max_duration=0.80,
        yawning_threshold=1.0,
        phone_threshold=1.0,
        seatbelt_threshold=1.5,
        perclos_window=10.0,
        perclos_minimum_observation=10.0,
        perclos_warning_threshold=0.20,
        perclos_drowsy_threshold=0.30,
        eye_confidence_threshold=0.50,
        mouth_confidence_threshold=0.65,
        max_eye_observation_gap=0.80,
        yawning_missing_grace_period=0.80,
        phone_missing_grace_period=0.80,
        seatbelt_box_grace_period=0.80,
    ):
        self.eye_closure_threshold = float(eye_closure_threshold)
        self.warning_closure_threshold = float(warning_closure_threshold)
        self.blink_max_duration = float(blink_max_duration)
        self.yawning_threshold = max(1.2, float(yawning_threshold))
        self.phone_threshold = float(phone_threshold)
        self.seatbelt_threshold = float(seatbelt_threshold)

        self.perclos_window = float(perclos_window)
        self.perclos_minimum_observation = float(perclos_minimum_observation)
        self.perclos_warning_threshold = float(perclos_warning_threshold)
        self.perclos_drowsy_threshold = float(perclos_drowsy_threshold)

        self.eye_confidence_threshold = float(eye_confidence_threshold)
        self.mouth_confidence_threshold = float(mouth_confidence_threshold)
        self.max_eye_observation_gap = float(max_eye_observation_gap)
        self.yawning_missing_grace_period = float(yawning_missing_grace_period)
        self.phone_missing_grace_period = float(phone_missing_grace_period)
        self.seatbelt_box_grace_period = float(seatbelt_box_grace_period)

        # Eye state
        self.eyes_closed_since = None
        self.last_eye_state = None
        self.last_eye_timestamp = None
        self.current_closure_duration = 0.0
        self.blink_count = 0
        self.completed_closures = deque(maxlen=100)

        # PERCLOS: (timestamp, bilateral_closed)
        self.eye_history = deque()

        # Yawning
        self.yawning_since = None
        self.last_yawning_timestamp = None
        self.last_yawning_confidence = 0.0
        self.last_mouth_observation_id = None

        # Phone
        self.phone_detected_since = None
        self.last_phone_timestamp = None
        self.last_phone_confidence = 0.0
        self.last_phone_box = None

        # Seatbelt
        self.seatbelt_missing_since = None
        self.last_seatbelt_timestamp = None
        self.last_seatbelt_confidence = 0.0
        self.last_seatbelt_box = None

        self.last_timestamp = None

    # ------------------------------------------------------------------
    # Eye helpers
    # ------------------------------------------------------------------
    def _get_valid_eyes(self, eyes):
        """Return usable eye predictions without discarding moderate CNN confidence."""
        valid = []
        for eye in eyes or []:
            state = eye.get("state")
            confidence = float(eye.get("confidence", 0.0))
            if state not in {"OPEN", "CLOSED"}:
                continue
            if confidence < self.eye_confidence_threshold:
                continue
            valid.append({
                "state": state,
                "confidence": confidence,
                "closed_probability": float(eye.get("closed_probability", 1.0 if state == "CLOSED" else 0.0)),
                "open_probability": float(eye.get("open_probability", 1.0 if state == "OPEN" else 0.0)),
                "yolo_class": eye.get("yolo_class"),
                "box": eye.get("box"),
            })
        return valid

    def _get_eye_probabilities(self, eyes):
        valid = self._get_valid_eyes(eyes)
        if len(valid) < 2:
            return None, None

        closed_probs = []
        open_probs = []
        for item in valid:
            c = item["closed_probability"]
            o = item["open_probability"]
            # If the CNN is only moderately confident, retain the YOLO
            # class as a weak localization/classification prior. YOLO was
            # trained with explicit open_eye/closed_eye classes.
            if item["confidence"] < 0.65 and item.get("yolo_class") in {0, 1}:
                yolo_closed = 1.0 if int(item["yolo_class"]) == 1 else 0.0
                yolo_open = 1.0 - yolo_closed
                c = 0.70 * c + 0.30 * yolo_closed
                o = 0.70 * o + 0.30 * yolo_open
            closed_probs.append(c)
            open_probs.append(o)
        return sum(closed_probs) / len(closed_probs), sum(open_probs) / len(open_probs)

    def _get_eye_state(self, eyes):
        closed_probability, open_probability = self._get_eye_probabilities(eyes)
        if closed_probability is None:
            return None
        if closed_probability >= 0.60 and closed_probability > open_probability:
            return "CLOSED"
        if open_probability >= 0.60 and open_probability > closed_probability:
            return "OPEN"
        return None

    def _get_display_eye_state(self, eyes):
        valid = self._get_valid_eyes(eyes)
        closed_probability, open_probability = self._get_eye_probabilities(eyes)
        if closed_probability is not None:
            if closed_probability >= 0.60 and closed_probability > open_probability:
                return "CLOSED"
            if open_probability >= 0.60 and open_probability > closed_probability:
                return "OPEN"
        if valid:
            return max(valid, key=lambda item: item["confidence"])["state"]
        return "OPEN"

    # ------------------------------------------------------------------
    # PERCLOS
    # ------------------------------------------------------------------
    def _update_perclos(self, timestamp, eye_state):
        self.eye_history.append((timestamp, eye_state == "CLOSED"))
        window_start = timestamp - self.perclos_window
        while self.eye_history and self.eye_history[0][0] < window_start:
            self.eye_history.popleft()

        if len(self.eye_history) < 2:
            return 0.0

        closed_duration = 0.0
        observed_duration = 0.0
        history = list(self.eye_history)
        for i in range(len(history) - 1):
            t1, closed = history[i]
            t2, _ = history[i + 1]
            dt = max(0.0, t2 - t1)
            observed_duration += dt
            if closed:
                closed_duration += dt

        return closed_duration / observed_duration if observed_duration > 0 else 0.0

    def _observation_duration(self):
        if len(self.eye_history) < 2:
            return 0.0
        return self.eye_history[-1][0] - self.eye_history[0][0]

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, inference_result, timestamp=None):
        if timestamp is None:
            timestamp = time.monotonic()
        timestamp = float(timestamp)

        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Timestamp must not move backwards.")

        result = {
            "timestamp": timestamp,
            "eye_state": "UNKNOWN",
            "drowsiness": {
                "state": "NORMAL",
                "eye_closed_duration": self.current_closure_duration,
                "perclos": 0.0,
                "blink_count": self.blink_count,
                "average_closure_duration": 0.0,
                "alert": False,
            },
            "yawning": {"state": "NON_YAWNING", "duration": 0.0, "confidence": 0.0, "alert": False},
            "phone": {"state": "NOT_DETECTED", "duration": 0.0, "confidence": 0.0, "box": None, "alert": False},
            "seatbelt": {
                "state": "DETECTED", "duration": 0.0, "alert": False,
                "detected": False, "confidence": 0.0, "box": None,
                "display_detected": False, "display_confidence": 0.0, "display_box": None,
            },
        }

        # ==============================================================
        # EYES / DROWSINESS
        # ==============================================================
        eyes = inference_result.get("eyes", []) or []
        result["eye_state"] = self._get_display_eye_state(eyes)
        bilateral_state = self._get_eye_state(eyes)

        if bilateral_state is not None:
            perclos = self._update_perclos(timestamp, bilateral_state)
            result["drowsiness"]["perclos"] = perclos
            self.last_eye_state = bilateral_state
            self.last_eye_timestamp = timestamp

            if bilateral_state == "CLOSED":
                if self.eyes_closed_since is None:
                    self.eyes_closed_since = timestamp

                self.current_closure_duration = max(0.0, timestamp - self.eyes_closed_since)
                result["drowsiness"]["eye_closed_duration"] = self.current_closure_duration

                if self.current_closure_duration >= self.eye_closure_threshold:
                    result["drowsiness"]["state"] = "DROWSY"
                    result["drowsiness"]["alert"] = True
                else:
                    # Short eye closure is a blink/normal event, not a
                    # separate drowsiness state.
                    result["drowsiness"]["state"] = "NORMAL"

            else:  # OPEN
                if self.eyes_closed_since is not None:
                    closure_duration = max(0.0, timestamp - self.eyes_closed_since)
                    if 0.0 < closure_duration <= self.blink_max_duration:
                        self.blink_count += 1
                        self.completed_closures.append(closure_duration)

                self.eyes_closed_since = None
                self.current_closure_duration = 0.0
                result["drowsiness"]["state"] = "NORMAL"
                result["drowsiness"]["eye_closed_duration"] = 0.0

        else:
            # A short YOLO/eye-model dropout must not instantly destroy a
            # valid closure sequence. Reuse the last bilateral state only
            # inside a small, explicit grace period.
            gap = (
                timestamp - self.last_eye_timestamp
                if self.last_eye_timestamp is not None
                else float("inf")
            )

            if self.last_eye_state == "CLOSED" and gap <= self.max_eye_observation_gap and self.eyes_closed_since is not None:
                self.current_closure_duration = max(0.0, timestamp - self.eyes_closed_since)
                result["eye_state"] = "CLOSED"
                result["drowsiness"]["eye_closed_duration"] = self.current_closure_duration
                if self.current_closure_duration >= self.eye_closure_threshold:
                    result["drowsiness"]["state"] = "DROWSY"
                    result["drowsiness"]["alert"] = True
                else:
                    # Short eye closure is a blink/normal event, not a
                    # separate drowsiness state.
                    result["drowsiness"]["state"] = "NORMAL"
                # Do not add duplicate samples to PERCLOS while uncertain.
            elif self.last_eye_state == "OPEN" and gap <= self.max_eye_observation_gap:
                result["eye_state"] = "OPEN"
                result["drowsiness"]["state"] = "NORMAL"
            else:
                # No fresh bilateral observation: preserve the last safe state
                # rather than exposing UNKNOWN in the user-facing UI.
                result["drowsiness"]["state"] = "DROWSY" if (
                    self.last_eye_state == "CLOSED"
                    and self.eyes_closed_since is not None
                    and self.current_closure_duration >= self.eye_closure_threshold
                ) else "NORMAL"
                if self.last_eye_state == "CLOSED" and gap > self.max_eye_observation_gap:
                    self.eyes_closed_since = None
                    self.current_closure_duration = 0.0
                    self.last_eye_state = None
                    self.last_eye_timestamp = None

        # PERCLOS is a fatigue metric, not a sticky immediate alert.
        # It intentionally does not override current eye-closure state.

        # ==============================================================
        # YAWNING
        # ==============================================================
        mouth = inference_result.get("mouth", {}) or {}
        mouth_state = mouth.get("state", "NON_YAWNING")
        mouth_conf = float(mouth.get("confidence", 0.0))
        yawn_probability = float(mouth.get("yawning_probability", 1.0 if mouth_state == "YAWNING" else 0.0))
        observation_id = mouth.get("observation_id")
        fresh_mouth = bool(mouth.get("fresh_observation", True))

        # A cached mouth classification is NOT a new observation. This is
        # critical: otherwise one false YAWNING prediction gets replayed
        # every other frame and the temporal timer incorrectly turns it into
        # a sustained yawn.
        new_mouth_observation = (
            fresh_mouth
            and observation_id is not None
            and observation_id != self.last_mouth_observation_id
        )

        if new_mouth_observation:
            self.last_mouth_observation_id = observation_id

            valid_yawn = (
                mouth_state == "YAWNING"
                and mouth_conf >= self.mouth_confidence_threshold
                and yawn_probability >= 0.70
            )

            if valid_yawn:
                if self.yawning_since is None:
                    self.yawning_since = timestamp
                self.last_yawning_timestamp = timestamp
                self.last_yawning_confidence = mouth_conf
            else:
                # A fresh NON_YAWNING observation is authoritative and
                # immediately cancels a pending/active yawn.
                self.yawning_since = None
                self.last_yawning_timestamp = None
                self.last_yawning_confidence = 0.0

        # Report the temporal state from the current confirmed run.
        if self.yawning_since is not None and self.last_yawning_timestamp is not None:
            gap = max(0.0, timestamp - self.last_yawning_timestamp)
            if gap <= self.yawning_missing_grace_period:
                duration = max(0.0, timestamp - self.yawning_since)
                result["yawning"]["duration"] = duration
                result["yawning"]["confidence"] = self.last_yawning_confidence
                if duration >= self.yawning_threshold:
                    result["yawning"]["state"] = "YAWNING"
                    result["yawning"]["alert"] = True
                else:
                    result["yawning"]["state"] = "NON_YAWNING"
            else:
                self.yawning_since = None
                self.last_yawning_timestamp = None
                self.last_yawning_confidence = 0.0

        if self.yawning_since is None:
            result["yawning"]["state"] = "NON_YAWNING"
            result["yawning"]["confidence"] = yawn_probability if mouth_state == "YAWNING" else mouth_conf

        # ==============================================================
        # PHONE
        # ==============================================================
        phone = inference_result.get("phone", {}) or {}
        phone_detected = bool(phone.get("detected", False))

        if phone_detected:
            if self.phone_detected_since is None:
                self.phone_detected_since = timestamp
            self.last_phone_timestamp = timestamp
            self.last_phone_confidence = float(phone.get("confidence", 0.0))
            self.last_phone_box = phone.get("box")

        phone_gap = (
            timestamp - self.last_phone_timestamp
            if self.last_phone_timestamp is not None
            else float("inf")
        )
        phone_active = self.phone_detected_since is not None and phone_gap <= self.phone_missing_grace_period

        if phone_active:
            duration = max(0.0, timestamp - self.phone_detected_since)
            result["phone"]["duration"] = duration
            result["phone"]["confidence"] = self.last_phone_confidence
            result["phone"]["box"] = self.last_phone_box
            if duration >= self.phone_threshold:
                result["phone"]["state"] = "PHONE_USAGE"
                result["phone"]["alert"] = True
            else:
                result["phone"]["state"] = "PHONE_USAGE" if duration >= self.phone_threshold else "NOT_DETECTED"
        else:
            self.phone_detected_since = None
            self.last_phone_timestamp = None
            self.last_phone_confidence = 0.0
            self.last_phone_box = None
            result["phone"]["state"] = "NOT_DETECTED"

        # ==============================================================
        # SEATBELT
        # ==============================================================
        seatbelt = inference_result.get("seatbelt", {}) or {}
        seatbelt_detected = bool(seatbelt.get("detected", False))

        if seatbelt_detected:
            self.seatbelt_missing_since = None
            self.last_seatbelt_timestamp = timestamp
            self.last_seatbelt_confidence = float(seatbelt.get("confidence", 0.0))
            self.last_seatbelt_box = seatbelt.get("box")

            result["seatbelt"].update({
                "state": "DETECTED",
                "detected": True,
                "confidence": self.last_seatbelt_confidence,
                "box": self.last_seatbelt_box,
                "display_detected": True,
                "display_confidence": self.last_seatbelt_confidence,
                "display_box": self.last_seatbelt_box,
            })
        else:
            if self.seatbelt_missing_since is None:
                self.seatbelt_missing_since = timestamp
            missing_duration = max(0.0, timestamp - self.seatbelt_missing_since)
            result["seatbelt"]["duration"] = missing_duration

            box_gap = (
                timestamp - self.last_seatbelt_timestamp
                if self.last_seatbelt_timestamp is not None
                else float("inf")
            )
            if self.last_seatbelt_timestamp is not None and box_gap <= self.seatbelt_box_grace_period:
                result["seatbelt"].update({
                    "state": "DETECTED",
                    "confidence": self.last_seatbelt_confidence,
                    "box": self.last_seatbelt_box,
                    "display_detected": True,
                    "display_confidence": self.last_seatbelt_confidence,
                    "display_box": self.last_seatbelt_box,
                })
            elif missing_duration >= self.seatbelt_threshold:
                result["seatbelt"]["state"] = "VIOLATION"
                result["seatbelt"]["alert"] = True
            else:
                result["seatbelt"]["state"] = "DETECTED"

        # ==============================================================
        # Statistics
        # ==============================================================
        result["drowsiness"]["blink_count"] = self.blink_count
        if self.completed_closures:
            result["drowsiness"]["average_closure_duration"] = sum(self.completed_closures) / len(self.completed_closures)

        self.last_timestamp = timestamp
        return result

    def reset(self):
        self.eyes_closed_since = None
        self.last_eye_state = None
        self.last_eye_timestamp = None
        self.current_closure_duration = 0.0
        self.blink_count = 0
        self.completed_closures.clear()
        self.eye_history.clear()

        self.yawning_since = None
        self.last_yawning_timestamp = None
        self.last_yawning_confidence = 0.0
        self.last_mouth_observation_id = None

        self.phone_detected_since = None
        self.last_phone_timestamp = None
        self.last_phone_confidence = 0.0
        self.last_phone_box = None

        self.seatbelt_missing_since = None
        self.last_seatbelt_timestamp = None
        self.last_seatbelt_confidence = 0.0
        self.last_seatbelt_box = None

        self.last_timestamp = None

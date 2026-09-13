from collections import deque
import time


class DriverTemporalEngine:
    """
    Temporal reasoning layer for the Driver Monitoring System.

    Consumes frame-level predictions from:
        - YOLO
        - Eye CNN
        - Mouth CNN
        - MediaPipe

    Performs:
        - Eye-state tracking
        - Prolonged eye closure detection
        - Blink detection
        - PERCLOS measurement
        - Yawning persistence
        - Phone persistence
        - Seatbelt persistence
        - Confidence filtering
        - Missing-observation handling

    Important:
        This is a rule-based temporal reasoning layer.
        It is NOT a trained temporal neural network.
    """

    def __init__(
        self,
        eye_closure_threshold=2.0,
        warning_closure_threshold=1.0,
        blink_max_duration=0.80,
        yawning_threshold=1.5,
        phone_threshold=1.0,
        seatbelt_threshold=1.0,
        perclos_window=10.0,
        perclos_minimum_observation=10.0,
        perclos_warning_threshold=0.20,
        perclos_drowsy_threshold=0.30,
        eye_confidence_threshold=0.70,
        mouth_confidence_threshold=0.70,
        max_eye_observation_gap=0.75,
        phone_missing_grace_period=2.0,
        seatbelt_missing_threshold=2.5,
        seatbelt_box_grace_period=0.75,
    ):

        # ============================================================
        # THRESHOLDS
        # ============================================================

        self.eye_closure_threshold = eye_closure_threshold
        self.warning_closure_threshold = warning_closure_threshold

        # A closure shorter than this is considered a blink.
        self.blink_max_duration = blink_max_duration

        self.yawning_threshold = yawning_threshold
        self.phone_threshold = phone_threshold
        self.phone_missing_grace_period = phone_missing_grace_period
        # Legacy name retained for compatibility.
        # Seatbelt violation uses the more explicit missing threshold.
        self.seatbelt_threshold = seatbelt_threshold
        self.seatbelt_missing_threshold = seatbelt_missing_threshold
        self.seatbelt_box_grace_period = seatbelt_box_grace_period

        self.perclos_window = perclos_window
        self.perclos_minimum_observation = (
            perclos_minimum_observation
        )

        self.perclos_warning_threshold = (
            perclos_warning_threshold
        )

        self.perclos_drowsy_threshold = (
            perclos_drowsy_threshold
        )

        self.eye_confidence_threshold = (
            eye_confidence_threshold
        )

        self.mouth_confidence_threshold = (
            mouth_confidence_threshold
        )

        # If reliable eye observations disappear for longer than
        # this period, the previous continuous closure is considered
        # broken.
        self.max_eye_observation_gap = (
            max_eye_observation_gap
        )

        # ============================================================
        # EYE STATE
        # ============================================================

        self.eyes_closed_since = None

        self.last_eye_state = None
        self.last_eye_timestamp = None

        self.current_closure_duration = 0.0

        self.blink_count = 0

        self.completed_closures = []

        # ============================================================
        # PERCLOS HISTORY
        # ============================================================

        # Stores:
        #     (timestamp, is_closed)
        self.eye_history = deque()

        # ============================================================
        # YAWNING STATE
        # ============================================================

        self.yawning_since = None
        self.last_yawning_timestamp = None

        # ============================================================
        # PHONE STATE
        # ============================================================

        self.phone_detected_since = None
        self.last_phone_timestamp = None
        self.last_phone_confidence = 0.0
        self.last_phone_box = None

        # ============================================================
        # SEATBELT STATE
        # ============================================================

        self.seatbelt_missing_since = None
        self.last_seatbelt_timestamp = None

        # Last reliable seatbelt detection. Used to tolerate short
        # YOLO detector dropouts and to keep the last box visible
        # briefly in the UI.
        self.last_seatbelt_box = None
        self.last_seatbelt_confidence = 0.0

        # ============================================================
        # GLOBAL
        # ============================================================

        self.last_timestamp = None

    # ==================================================================
    # EYE STATE HELPERS
    # ==================================================================

    def _get_valid_eyes(self, eyes):
        """
        Return eye predictions that have a valid state and
        sufficient confidence.
        """

        if not eyes:
            return []

        valid_eyes = []

        for eye in eyes:

            state = eye.get("state")

            confidence = float(
                eye.get("confidence", 0.0)
            )

            if (
                state in {"OPEN", "CLOSED"}
                and confidence >= self.eye_confidence_threshold
            ):
                valid_eyes.append(
                    {
                        "state": state,
                        "confidence": confidence,
                    }
                )

        return valid_eyes

    def _get_eye_state(self, eyes):
        """
        Determine bilateral eye state for drowsiness decisions.

        Returns:
            "OPEN"
            "CLOSED"
            None

        IMPORTANT:
        Drowsiness requires bilateral evidence.

        If one eye is OPEN and the other is CLOSED,
        the result is considered uncertain.
        """

        valid_eyes = self._get_valid_eyes(eyes)

        if len(valid_eyes) < 2:
            return None

        states = [
            eye["state"]
            for eye in valid_eyes
        ]

        if all(
            state == "CLOSED"
            for state in states
        ):
            return "CLOSED"

        if all(
            state == "OPEN"
            for state in states
        ):
            return "OPEN"

        # One OPEN + one CLOSED
        return None

    def _get_display_eye_state(self, eyes):
        """
        Determine the eye state to display in the UI.

        Unlike the drowsiness decision, this method can use
        a single reliable eye prediction.

        Returns:
            "OPEN"
            "CLOSED"
            "UNKNOWN"
        """

        valid_eyes = self._get_valid_eyes(eyes)

        if not valid_eyes:
            return "UNKNOWN"

        # --------------------------------------------------------------
        # Two or more reliable eyes
        # --------------------------------------------------------------

        if len(valid_eyes) >= 2:

            states = [
                eye["state"]
                for eye in valid_eyes
            ]

            if all(
                state == "CLOSED"
                for state in states
            ):
                return "CLOSED"

            if all(
                state == "OPEN"
                for state in states
            ):
                return "OPEN"

            # Conflicting evidence.
            #
            # Use the eye with the highest confidence for display.
            strongest_eye = max(
                valid_eyes,
                key=lambda item: item["confidence"]
            )

            return strongest_eye["state"]

        # --------------------------------------------------------------
        # Only one reliable eye
        # --------------------------------------------------------------

        return valid_eyes[0]["state"]

    # ==================================================================
    # PERCLOS
    # ==================================================================

    def _update_perclos(
        self,
        timestamp,
        eye_state,
    ):
        """
        Update rolling PERCLOS.

        PERCLOS =
            time eyes are closed
            --------------------
            observed eye time
        """

        # Only valid bilateral observations enter the PERCLOS
        # calculation.
        self.eye_history.append(
            (
                timestamp,
                eye_state == "CLOSED",
            )
        )

        window_start = (
            timestamp - self.perclos_window
        )

        # Remove observations outside the rolling window.
        while (
            self.eye_history
            and self.eye_history[0][0] < window_start
        ):
            self.eye_history.popleft()

        if len(self.eye_history) < 2:
            return 0.0

        history = list(
            self.eye_history
        )

        closed_duration = 0.0
        observed_duration = 0.0

        for i in range(
            len(history) - 1
        ):

            t1, closed = history[i]
            t2, _ = history[i + 1]

            duration = t2 - t1

            if duration < 0:
                continue

            observed_duration += duration

            if closed:
                closed_duration += duration

        if observed_duration <= 0:
            return 0.0

        return (
            closed_duration
            / observed_duration
        )

    def _get_perclos_observation_duration(self):
        """
        Return the amount of eye observation time currently
        available for PERCLOS decisions.
        """

        if len(self.eye_history) < 2:
            return 0.0

        return (
            self.eye_history[-1][0]
            - self.eye_history[0][0]
        )

    # ==================================================================
    # MAIN UPDATE
    # ==================================================================

    def update(
        self,
        inference_result,
        timestamp=None,
    ):
        """
        Update temporal state using one frame of inference results.

        Parameters:
            inference_result:
                Result returned by DriverInferenceEngine.

            timestamp:
                Video timestamp in seconds.

                If None, time.monotonic() is used.

        Returns:
            Dictionary containing temporal states.
        """

        # --------------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------------

        if timestamp is None:
            timestamp = time.monotonic()

        if (
            self.last_timestamp is not None
            and timestamp < self.last_timestamp
        ):
            raise ValueError(
                "Timestamp must not move backwards."
            )

        # ==============================================================
        # INITIAL RESULT
        # ==============================================================

        result = {
            "timestamp": timestamp,

            "drowsiness": {
                "state": "NORMAL",

                # Current continuous eye closure duration.
                "eye_closed_duration": 0.0,

                # Rolling PERCLOS.
                "perclos": 0.0,

                # Number of completed short closures.
                "blink_count": self.blink_count,

                # Average duration of completed short closures.
                "average_closure_duration": 0.0,

                "alert": False,
            },

            # Separate eye state.
            #
            # This is useful for UI because:
            #
            # EYES: CLOSED
            # DRIVER STATUS: UNSAFE
            #
            # are different concepts.
            "eye_state": "UNKNOWN",

            "yawning": {
                "state": "UNKNOWN",
                "duration": 0.0,
                "alert": False,
            },

            "phone": {
                "state": "NOT_DETECTED",
                "duration": 0.0,
                "alert": False,
            },

            "seatbelt": {
                "state": "UNKNOWN",
                "duration": 0.0,
                "alert": False,
                "detected": False,
                "confidence": 0.0,
                "box": None,
                "display_detected": False,
                "display_confidence": 0.0,
                "display_box": None,
            },
        }

        # ==============================================================
        # EYE ANALYSIS
        # ==============================================================

        eyes = inference_result.get(
            "eyes",
            []
        )

        # --------------------------------------------------------------
        # Display eye state
        # --------------------------------------------------------------

        display_eye_state = (
            self._get_display_eye_state(
                eyes
            )
        )

        result["eye_state"] = (
            display_eye_state
        )

        # --------------------------------------------------------------
        # Bilateral eye state for drowsiness
        # --------------------------------------------------------------

        eye_state = self._get_eye_state(
            eyes
        )

        # ==============================================================
        # VALID BILATERAL EYE OBSERVATION
        # ==============================================================

        if eye_state is not None:

            # ----------------------------------------------------------
            # PERCLOS
            # ----------------------------------------------------------

            perclos = self._update_perclos(
                timestamp,
                eye_state,
            )

            result[
                "drowsiness"
            ]["perclos"] = perclos

            # ----------------------------------------------------------
            # EYES CLOSED
            # ----------------------------------------------------------

            if eye_state == "CLOSED":

                # Start a new closure sequence.
                if self.eyes_closed_since is None:

                    self.eyes_closed_since = (
                        timestamp
                    )

                self.current_closure_duration = (
                    timestamp
                    - self.eyes_closed_since
                )

                result[
                    "drowsiness"
                ]["eye_closed_duration"] = (
                    self.current_closure_duration
                )

                # ------------------------------------------------------
                # DROWSINESS DECISION
                # ------------------------------------------------------

                if (
                    self.current_closure_duration
                    >= self.eye_closure_threshold
                ):

                    result[
                        "drowsiness"
                    ]["state"] = "DROWSY"

                    result[
                        "drowsiness"
                    ]["alert"] = True

                elif (
                    self.current_closure_duration
                    >= self.warning_closure_threshold
                ):

                    result[
                        "drowsiness"
                    ]["state"] = "WARNING"

                else:

                    result[
                        "drowsiness"
                    ]["state"] = "EYES_CLOSED"

            # ----------------------------------------------------------
            # EYES OPEN
            # ----------------------------------------------------------

            elif eye_state == "OPEN":

                # If a previous closure exists, determine whether
                # it was a blink.
                if self.eyes_closed_since is not None:

                    closure_duration = (
                        timestamp
                        - self.eyes_closed_since
                    )

                    # Only genuinely short closures count as blinks.
                    if (
                        closure_duration
                        <= self.blink_max_duration
                    ):

                        self.blink_count += 1

                        self.completed_closures.append(
                            closure_duration
                        )

                        # Keep only the most recent 100 closures.
                        if (
                            len(
                                self.completed_closures
                            ) > 100
                        ):
                            self.completed_closures.pop(
                                0
                            )

                # Reset continuous closure.
                self.eyes_closed_since = None

                self.current_closure_duration = 0.0

                result[
                    "drowsiness"
                ]["state"] = "NORMAL"

                result[
                    "drowsiness"
                ]["eye_closed_duration"] = 0.0

            # ----------------------------------------------------------
            # Save eye observation timestamp
            # ----------------------------------------------------------

            self.last_eye_state = eye_state

            self.last_eye_timestamp = timestamp

        # ==============================================================
        # EYE OBSERVATION UNCERTAIN
        # ==============================================================

        else:

            result[
                "drowsiness"
            ]["state"] = "UNKNOWN"

            # Keep the last known duration visible temporarily.
            result[
                "drowsiness"
            ]["eye_closed_duration"] = (
                self.current_closure_duration
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT automatically extend a closure based on missing
            # eye observations.
            #
            # If the gap becomes too large, break the closure sequence.
            # ----------------------------------------------------------

            if (
                self.last_eye_timestamp is not None
                and (
                    timestamp
                    - self.last_eye_timestamp
                ) > self.max_eye_observation_gap
            ):

                self.eyes_closed_since = None

                self.current_closure_duration = 0.0

                self.last_eye_state = None

                result[
                    "drowsiness"
                ]["eye_closed_duration"] = 0.0

        # ==============================================================
        # PERCLOS / FATIGUE INDICATOR
        # ==============================================================

        # PERCLOS is retained as a rolling fatigue metric and is shown
        # in the UI, but it must NOT keep the driver in DROWSY state
        # after the eyes have reopened. A previous long closure can keep
        # PERCLOS high for several seconds by definition.
        #
        # DROWSINESS_ALERT is therefore generated only by the direct
        # continuous bilateral-eye-closure rule above. This prevents the
        # false situation: EYES=OPEN + DROWSINESS=DROWSY + ALERT=ON.
        #
        # The thresholds remain available for future fatigue-risk logic
        # that can be introduced separately from the immediate alert.

        # ==============================================================
        # YAWNING
        # ==============================================================

        mouth = inference_result.get(
            "mouth",
            {}
        )

        mouth_state = mouth.get(
            "state",
            "UNKNOWN"
        )

        mouth_confidence = float(
            mouth.get(
                "confidence",
                0.0
            )
        )

        # --------------------------------------------------------------
        # Valid yawning signal
        # --------------------------------------------------------------

        if (
            mouth_state == "YAWNING"
            and mouth_confidence
            >= self.mouth_confidence_threshold
        ):

            if self.yawning_since is None:

                self.yawning_since = (
                    timestamp
                )

            self.last_yawning_timestamp = (
                timestamp
            )

            duration = (
                timestamp
                - self.yawning_since
            )

            result[
                "yawning"
            ]["duration"] = duration

            if (
                duration
                >= self.yawning_threshold
            ):

                result[
                    "yawning"
                ]["state"] = "YAWNING"

                result[
                    "yawning"
                ]["alert"] = True

            else:

                result[
                    "yawning"
                ]["state"] = "YAWNING_PENDING"

        # --------------------------------------------------------------
        # Explicit non-yawning signal
        # --------------------------------------------------------------

        elif mouth_state == "NON_YAWNING":

            self.yawning_since = None

            self.last_yawning_timestamp = (
                timestamp
            )

            result[
                "yawning"
            ]["state"] = "NOT_YAWNING"

        # --------------------------------------------------------------
        # Unknown / low-confidence signal
        # --------------------------------------------------------------

        else:

            result[
                "yawning"
            ]["state"] = "UNKNOWN"

        # ==============================================================
        # PHONE
        # ==============================================================

        phone = inference_result.get(
            "phone",
            {}
        )

        phone_detected = bool(
            phone.get("detected", False)
        )

        if phone_detected:

            # Start a new phone-presence sequence only when there is
            # no active sequence.
            if self.phone_detected_since is None:
                self.phone_detected_since = timestamp

            self.last_phone_timestamp = timestamp
            self.last_phone_confidence = float(
                phone.get("confidence", 0.0)
            )
            self.last_phone_box = phone.get("box")

            duration = (
                timestamp
                - self.phone_detected_since
            )

            result["phone"]["duration"] = duration

            if duration >= self.phone_threshold:
                result["phone"]["state"] = "PHONE_USAGE"
                result["phone"]["alert"] = True
            else:
                result["phone"]["state"] = "PHONE_PENDING"

        else:

            # YOLO can intermittently miss a phone while it remains
            # visible. Keep the active sequence alive briefly after the
            # last confirmed detection instead of resetting immediately.
            within_grace = (
                self.phone_detected_since is not None
                and self.last_phone_timestamp is not None
                and (
                    timestamp - self.last_phone_timestamp
                    <= self.phone_missing_grace_period
                )
            )

            if within_grace:

                duration = (
                    timestamp
                    - self.phone_detected_since
                )

                result["phone"]["duration"] = duration

                if duration >= self.phone_threshold:
                    result["phone"]["state"] = "PHONE_USAGE"
                    result["phone"]["alert"] = True
                else:
                    result["phone"]["state"] = "PHONE_PENDING"

            else:

                self.phone_detected_since = None
                self.last_phone_timestamp = None
                self.last_phone_confidence = 0.0
                self.last_phone_box = None

                result["phone"]["state"] = "NOT_DETECTED"

        # ==============================================================
        # SEATBELT
        # ==============================================================
        #
        # YOLO can temporarily lose the seatbelt when the driver moves.
        # A short detector dropout must NOT immediately become a
        # seatbelt violation.
        #
        # Two separate concepts are maintained:
        #   1. Safety state: violation only after sustained absence.
        #   2. Display state: keep the last reliable box visible briefly.
        # ==============================================================

        seatbelt = inference_result.get(
            "seatbelt",
            {}
        )

        seatbelt_detected = bool(
            seatbelt.get("detected", False)
        )

        seatbelt_confidence = float(
            seatbelt.get("confidence", 0.0)
        )

        seatbelt_box = seatbelt.get("box")

        # --------------------------------------------------------------
        # Seatbelt detected
        # --------------------------------------------------------------

        if seatbelt_detected:

            self.seatbelt_missing_since = None
            self.last_seatbelt_timestamp = timestamp

            self.last_seatbelt_box = seatbelt_box
            self.last_seatbelt_confidence = seatbelt_confidence

            result["seatbelt"]["state"] = "DETECTED"
            result["seatbelt"]["duration"] = 0.0
            result["seatbelt"]["detected"] = True
            result["seatbelt"]["confidence"] = seatbelt_confidence
            result["seatbelt"]["box"] = seatbelt_box

            result["seatbelt"]["display_detected"] = True
            result["seatbelt"]["display_confidence"] = (
                seatbelt_confidence
            )
            result["seatbelt"]["display_box"] = seatbelt_box

        # --------------------------------------------------------------
        # Seatbelt not detected on this frame
        # --------------------------------------------------------------

        else:

            if self.seatbelt_missing_since is None:
                self.seatbelt_missing_since = timestamp

            duration = (
                timestamp
                - self.seatbelt_missing_since
            )

            result["seatbelt"]["duration"] = duration
            result["seatbelt"]["detected"] = False

            # ----------------------------------------------------------
            # Short YOLO dropout
            # ----------------------------------------------------------
            #
            # Keep the last known seatbelt box visible for a short
            # grace period. This is presentation/state smoothing only;
            # it does not fabricate a new detection.
            # ----------------------------------------------------------

            if (
                self.last_seatbelt_timestamp is not None
                and (
                    timestamp
                    - self.last_seatbelt_timestamp
                ) <= self.seatbelt_box_grace_period
            ):

                result["seatbelt"]["state"] = "DETECTED"
                result["seatbelt"]["confidence"] = (
                    self.last_seatbelt_confidence
                )
                result["seatbelt"]["box"] = (
                    self.last_seatbelt_box
                )

                result["seatbelt"]["display_detected"] = True
                result["seatbelt"]["display_confidence"] = (
                    self.last_seatbelt_confidence
                )
                result["seatbelt"]["display_box"] = (
                    self.last_seatbelt_box
                )

            # ----------------------------------------------------------
            # Longer absence but not yet a violation
            # ----------------------------------------------------------

            elif duration < self.seatbelt_missing_threshold:

                result["seatbelt"]["state"] = "PENDING"

                result["seatbelt"]["display_detected"] = False

            # ----------------------------------------------------------
            # Sustained absence -> actual violation
            # ----------------------------------------------------------

            else:

                result["seatbelt"]["state"] = "VIOLATION"

                result["seatbelt"]["alert"] = True

                result["seatbelt"]["display_detected"] = False

        # ==============================================================
        # ==============================================================
        # BLINK STATISTICS
        # ==============================================================

        result[
            "drowsiness"
        ]["blink_count"] = (
            self.blink_count
        )

        if self.completed_closures:

            result[
                "drowsiness"
            ]["average_closure_duration"] = (
                sum(
                    self.completed_closures
                )
                / len(
                    self.completed_closures
                )
            )

        else:

            result[
                "drowsiness"
            ]["average_closure_duration"] = 0.0

        # ==============================================================
        # SAVE TIMESTAMP
        # ==============================================================

        self.last_timestamp = timestamp

        return result

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self):
        """
        Reset all temporal state.
        """

        # --------------------------------------------------------------
        # Eye state
        # --------------------------------------------------------------

        self.eyes_closed_since = None
        self.last_eye_state = None
        self.last_eye_timestamp = None

        self.current_closure_duration = 0.0

        self.blink_count = 0

        self.completed_closures.clear()

        # --------------------------------------------------------------
        # PERCLOS
        # --------------------------------------------------------------

        self.eye_history.clear()

        # --------------------------------------------------------------
        # Yawning
        # --------------------------------------------------------------

        self.yawning_since = None
        self.last_yawning_timestamp = None

        # --------------------------------------------------------------
        # Phone
        # --------------------------------------------------------------

        self.phone_detected_since = None
        self.last_phone_timestamp = None
        self.last_phone_confidence = 0.0
        self.last_phone_box = None

        # --------------------------------------------------------------
        # Seatbelt
        # --------------------------------------------------------------

        self.seatbelt_missing_since = None
        self.last_seatbelt_timestamp = None
        self.last_seatbelt_box = None
        self.last_seatbelt_confidence = 0.0

        # --------------------------------------------------------------
        # Global
        # --------------------------------------------------------------

        self.last_timestamp = None
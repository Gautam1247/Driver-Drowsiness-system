"""
Central Safety Engine

Combines inference and temporal-analysis results into one
unified driver safety decision.

This module is deterministic and does not perform machine learning.
"""


class CentralSafetyEngine:
    """
    Central decision layer of the Driver Monitoring System.
    """

    def __init__(self):

        # Internal risk scores.
        self.risk_scores = {
            "NORMAL": 0,
            "YAWNING": 1,
            "SEATBELT_VIOLATION": 2,
            "PHONE_USAGE": 3,
            "DROWSY": 4,
        }

        # Priority used when multiple violations occur.
        self.priority_order = [
            "DROWSINESS",
            "PHONE_USAGE",
            "SEATBELT_VIOLATION",
            "YAWNING",
        ]

    # ==============================================================
    # DROWSINESS
    # ==============================================================

    def _get_drowsiness_state(self, temporal_result):
        """
        Extract drowsiness information from Temporal Engine V2.

        Actual Temporal Engine structure:

        temporal_result["drowsiness"] = {
            "state": ...,
            "eye_closed_duration": ...,
            "perclos": ...,
            "blink_count": ...,
            "average_closure_duration": ...,
            "alert": ...
        }
        """

        drowsiness = temporal_result.get(
            "drowsiness",
            {}
        )

        state = drowsiness.get(
            "state",
            "UNKNOWN"
        )

        return {
            "active": state == "DROWSY",

            "warning": state == "WARNING",

            "state": state,

            "closure_duration": drowsiness.get(
                "eye_closed_duration",
                0.0
            ),

            "perclos": drowsiness.get(
                "perclos",
                0.0
            ),

            "blink_count": drowsiness.get(
                "blink_count",
                0
            ),

            "average_closure_duration": drowsiness.get(
                "average_closure_duration",
                0.0
            ),
        }

    # ==============================================================
    # YAWNING
    # ==============================================================

    def _get_yawning_state(self, temporal_result):

        yawning = temporal_result.get(
            "yawning",
            {}
        )

        state = yawning.get(
            "state",
            "UNKNOWN"
        )

        return {
            "active": state == "YAWNING",

            "pending": state == "YAWNING_PENDING",

            "state": state,

            "duration": yawning.get(
                "duration",
                0.0
            ),

            "confidence": yawning.get(
                "confidence",
                0.0
            ),
        }

    # ==============================================================
    # PHONE
    # ==============================================================

    def _get_phone_state(self, temporal_result):

        phone = temporal_result.get(
            "phone",
            {}
        )

        state = phone.get(
            "state",
            "UNKNOWN"
        )

        return {
            "active": state == "PHONE_USAGE",

            "pending": state == "PHONE_PENDING",

            "state": state,

            "duration": phone.get(
                "duration",
                0.0
            ),

            "confidence": phone.get(
                "confidence",
                0.0
            ),
        }

    # ==============================================================
    # SEATBELT
    # ==============================================================

    def _get_seatbelt_state(self, temporal_result):

        seatbelt = temporal_result.get(
            "seatbelt",
            {}
        )

        state = seatbelt.get(
            "state",
            "UNKNOWN"
        )

        return {
            "active": state == "VIOLATION",

            "pending": state == "PENDING",

            "state": state,

            "duration": seatbelt.get(
                "duration",
                0.0
            ),

            "confidence": seatbelt.get(
                "confidence",
                0.0
            ),
        }

    # ==============================================================
    # RISK CALCULATION
    # ==============================================================

    def _calculate_risk(
        self,
        drowsiness,
        yawning,
        phone,
        seatbelt
    ):
        """
        Calculate overall driver risk.

        Rules:

        Drowsiness + Phone
            -> CRITICAL

        Drowsiness
            -> HIGH

        Phone
            -> HIGH

        Seatbelt
            -> MODERATE

        Yawning
            -> LOW

        Warning / pending conditions
            -> lower temporary risk
        """

        # ----------------------------------------------------------
        # CRITICAL COMBINATION
        # ----------------------------------------------------------

        if (
            drowsiness["active"]
            and phone["active"]
        ):
            return "CRITICAL", 5

        # ----------------------------------------------------------
        # HIGH
        # ----------------------------------------------------------

        if drowsiness["active"]:
            return "HIGH", 4

        if phone["active"]:
            return "HIGH", 3

        # ----------------------------------------------------------
        # MODERATE
        # ----------------------------------------------------------

        if seatbelt["active"]:
            return "MODERATE", 2

        if drowsiness["warning"]:
            return "MODERATE", 2

        # ----------------------------------------------------------
        # LOW
        # ----------------------------------------------------------

        if yawning["active"]:
            return "LOW", 1

        if (
            yawning["pending"]
            or phone["pending"]
            or seatbelt["pending"]
        ):
            return "LOW", 1

        # ----------------------------------------------------------
        # NORMAL
        # ----------------------------------------------------------

        return "NORMAL", 0

    # ==============================================================
    # MAIN UPDATE FUNCTION
    # ==============================================================

    def update(
        self,
        inference_result,
        temporal_result
    ):
        """
        Combine inference and temporal results.

        Parameters
        ----------
        inference_result : dict
            Output from DriverInferenceEngine.process_frame()

        temporal_result : dict
            Output from DriverTemporalEngine.update()

        Returns
        -------
        dict
            Unified safety result.
        """

        # ----------------------------------------------------------
        # Extract states
        # ----------------------------------------------------------

        drowsiness = self._get_drowsiness_state(
            temporal_result
        )

        yawning = self._get_yawning_state(
            temporal_result
        )

        phone = self._get_phone_state(
            temporal_result
        )

        seatbelt = self._get_seatbelt_state(
            temporal_result
        )

        # ----------------------------------------------------------
        # Active violations
        # ----------------------------------------------------------

        active_violations = []

        if drowsiness["active"]:
            active_violations.append(
                "DROWSINESS"
            )

        if phone["active"]:
            active_violations.append(
                "PHONE_USAGE"
            )

        if seatbelt["active"]:
            active_violations.append(
                "SEATBELT_VIOLATION"
            )

        if yawning["active"]:
            active_violations.append(
                "YAWNING"
            )

        # ----------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------

        warnings = []

        if drowsiness["warning"]:
            warnings.append(
                "DROWSINESS_WARNING"
            )

        if yawning["pending"]:
            warnings.append(
                "YAWNING_PENDING"
            )

        if phone["pending"]:
            warnings.append(
                "PHONE_PENDING"
            )

        if seatbelt["pending"]:
            warnings.append(
                "SEATBELT_PENDING"
            )

        # ----------------------------------------------------------
        # Risk
        # ----------------------------------------------------------

        risk_level, risk_score = self._calculate_risk(
            drowsiness,
            yawning,
            phone,
            seatbelt
        )

        # ----------------------------------------------------------
        # Primary violation
        # ----------------------------------------------------------

        primary_violation = None

        for violation in self.priority_order:

            if violation in active_violations:

                primary_violation = violation

                break

        # ----------------------------------------------------------
        # Alerts
        # ----------------------------------------------------------

        alerts = []

        if drowsiness["active"]:

            alerts.append({
                "type": "DROWSINESS_ALERT",
                "severity": "CRITICAL",
                "message": (
                    "Driver appears drowsy. "
                    "Please stay alert."
                )
            })

        if phone["active"]:

            alerts.append({
                "type": "PHONE_ALERT",
                "severity": "HIGH",
                "message": (
                    "Phone usage detected "
                    "while driving."
                )
            })

        if seatbelt["active"]:

            alerts.append({
                "type": "SEATBELT_ALERT",
                "severity": "HIGH",
                "message": (
                    "Seatbelt violation detected."
                )
            })

        if yawning["active"]:

            alerts.append({
                "type": "YAWNING_ALERT",
                "severity": "LOW",
                "message": (
                    "Repeated yawning detected."
                )
            })

        if drowsiness["warning"]:

            alerts.append({
                "type": "DROWSINESS_WARNING",
                "severity": "MEDIUM",
                "message": (
                    "Eyes have remained closed "
                    "for an extended period."
                )
            })

        # ----------------------------------------------------------
        # Alert flag
        # ----------------------------------------------------------

        alert_active = (
            len(alerts) > 0
        )

        # ----------------------------------------------------------
        # Driver status
        # ----------------------------------------------------------

        if risk_level == "CRITICAL":

            driver_status = "CRITICAL"

        elif risk_level == "HIGH":

            driver_status = "UNSAFE"

        elif risk_level == "MODERATE":

            driver_status = "WARNING"

        elif risk_level == "LOW":

            driver_status = "CAUTION"

        else:

            driver_status = "NORMAL"

        # ----------------------------------------------------------
        # Unified result
        # ----------------------------------------------------------

        return {

            "driver_status": driver_status,

            "risk": {
                "level": risk_level,
                "score": risk_score,
            },

            "violations": {

                "drowsiness": drowsiness,

                "yawning": yawning,

                "phone_usage": phone,

                "seatbelt_violation": seatbelt,
            },

            "active_violations": active_violations,

            "warnings": warnings,

            "primary_violation": primary_violation,

            "alerts": alerts,

            "alert_active": alert_active,

            "inference": inference_result,
        }

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self):
        """
        Reset the central safety engine.

        The current engine is stateless, so this is provided
        for interface consistency and future extensions.
        """

        pass
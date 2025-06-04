from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from enum import Enum
import os
import asyncio
import json
import random




class DifficultyLevel(Enum):
    VERY_EASY = 1
    EASY = 2
    MODERATE = 3
    HARD = 4
    VERY_HARD = 5



def clamp_val(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float `x` to the range [min_val, max_val]."""
    return max(min_val, min(max_val, x))


@dataclass
class LessonPerformance:
    """
    Stores performance metrics for a single lesson, identified by:
      - lesson_id: a unique string (e.g., "1_2_3" for unitIndex=1, topicIndex=2, lessonIndex=3)
      - lesson_title: human-readable title
      - lesson_spec: specification of the lesson

    Fields:
        mastery_score: float in [0.0, 1.0]
        confidence_score: float in [0.0, 1.0]
        completion_rate: float in [0.0, 1.0]
        total_attempts: int >= 0
        total_time_spent: int (minutes)
        last_attempt: Optional[int] (session index, 1-based)
        retry_frequency: float in [0.0, 1.0]
        help_requests: int >= 0
        hint_usage_rate: float in [0.0, 1.0]
    """
    lesson_id: str
    lesson_title: str
    lesson_spec: str

    # Mastery & Confidence
    mastery_score: float = 0.0       # Value between 0.0 and 1.0
    confidence_score: float = 0.0    # Value between 0.0 and 1.0

    # Engagement Metrics
    completion_rate: float = 0.0     # Fraction between 0.0 and 1.0
    total_attempts: int = 0
    total_time_spent: int = 0        # In minutes

    # Instead of a datetime, store the index of the last session/attempt
    last_attempt: Optional[int] = None  # Session index (e.g., 1-based)

    # Learning Behavior
    retry_frequency: float = 0.0     # Fraction between 0.0 and 1.0
    help_requests: int = 0
    hint_usage_rate: float = 0.0     # Fraction between 0.0 and 1.0

    def get_engagement_score(self) -> float:
        """
        Calculate an engagement score for this lesson based on:
          1. Time spent (capped at 30 minutes → factor in [0,1]),
          2. Completion rate (already a fraction),
          3. Retry frequency (inverted, since more retries → less engagement).

        Returns:
            A float between 0.0 (no engagement) and 1.0 (high engagement).
        """
        if self.total_attempts == 0:
            return 0.0

        time_factor = min(self.total_time_spent / 30.0, 1.0)
        completion_factor = clamp_val(self.completion_rate)
        retry_factor = 1.0 - clamp_val(self.retry_frequency / 2.0)

        return (time_factor + completion_factor + retry_factor) / 3.0

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize this LessonPerformance instance to a dictionary of primitives,
        suitable for JSON encoding. Includes lesson_id, lesson_title, and lesson_spec.

        Returns:
            A dict containing all simple attributes.
        """
        return {
            "lesson_id": self.lesson_id,
            "lesson_title": self.lesson_title,
            "lesson_spec": self.lesson_spec,
            "mastery_score": self.mastery_score,
            "confidence_score": self.confidence_score,
            "completion_rate": self.completion_rate,
            "total_attempts": self.total_attempts,
            "total_time_spent": self.total_time_spent,
            "last_attempt": self.last_attempt,
            "retry_frequency": self.retry_frequency,
            "help_requests": self.help_requests,
            "hint_usage_rate": self.hint_usage_rate,
        }

    @staticmethod
    def from_json_dict(data: Dict[str, Any]) -> "LessonPerformance":
        """
        Reconstruct a LessonPerformance instance from a JSON-like dictionary.

        Args:
            data: A dictionary containing keys matching those from to_json_dict().

        Returns:
            A LessonPerformance object with fields populated from `data`.
        """
        lp = LessonPerformance(
            lesson_id=data["lesson_id"],
            lesson_title=data["lesson_title"],
            lesson_spec=data["lesson_spec"],
        )

        lp.mastery_score = data.get("mastery_score", 0.0)
        lp.confidence_score = data.get("confidence_score", 0.0)
        lp.completion_rate = data.get("completion_rate", 0.0)
        lp.total_attempts = data.get("total_attempts", 0)
        lp.total_time_spent = data.get("total_time_spent", 0)
        lp.last_attempt = data.get("last_attempt", None)
        lp.retry_frequency = data.get("retry_frequency", 0.0)
        lp.help_requests = data.get("help_requests", 0)
        lp.hint_usage_rate = data.get("hint_usage_rate", 0.0)

        return lp

    def generate_random_demo_data(
        profile: Any,
        curriculum: List[Dict[str, Any]],
        student_characteristics: Dict[str, float],
        *,
        max_unit_index: Optional[int] = None,
        max_topic_index: Optional[int] = None,
        max_lessons_per_session: int = 5
    ) -> None:
        """
        Populate `profile.lesson_performances` with demo data, but allows:
          1. Limiting generation to units up to max_unit_index (inclusive).
          2. Limiting generation to topics up to max_topic_index (inclusive) per unit.
          3. Controlling how many lessons appear in each simulated session.

        Instead of using lesson “attempt” counts as session indices, this
        method partitions the filtered lesson set into sessions of size
        `max_lessons_per_session`. Each lesson in a given partition receives
        the same session index (1-based).

        Args:
            profile: An object containing `lesson_performances: Dict[str, LessonPerformance]`.
            curriculum: A list of unit dicts, each with:
                - "unitIndex": int
                - "unitTitle": str
                - "topics": list of topic dicts, each with:
                    • "topicIndex": int
                    • "topicTitle": str
                    • "lessons": list of lesson dicts, each with:
                        ◦ "lessonIndex": int
                        ◦ "lessonTitle": str
                        ◦ "spec_content": str
            student_characteristics: Dict with keys:
                - "motivation": float in [0.0, 1.0]
                - "prior_knowledge": float in [0.0, 1.0]
                - "attention_span": float in [0.0, 1.0]
            max_unit_index: If set, only units with unitIndex <= this value are used.
            max_topic_index: If set, only topics with topicIndex <= this value are used.
            max_lessons_per_session: Maximum number of lessons to assign to each simulated session.

        Raises:
            ValueError: if required student_characteristics keys are missing.
        """
        # Validate student_characteristics
        required_keys = ["motivation", "prior_knowledge", "attention_span"]
        if not all(k in student_characteristics for k in required_keys):
            raise ValueError(f"Need {required_keys} in student_characteristics")

        mot = clamp_val(student_characteristics["motivation"])
        prior = clamp_val(student_characteristics["prior_knowledge"])
        att = clamp_val(student_characteristics["attention_span"])
        base_noise = (1.0 - att) * 0.3

        # 1. Collect all lesson metadata that meet unit/topic thresholds
        filtered_lessons = []
        for unit in curriculum:
            uidx = unit.get("unitIndex", 0)
            if max_unit_index is not None and uidx > max_unit_index:
                continue

            for topic in unit.get("topics", []):
                tidx = topic.get("topicIndex", 0)
                if max_topic_index is not None and tidx > max_topic_index:
                    continue

                for lesson in topic.get("lessons", []):
                    lesson_idx = lesson["lessonIndex"]
                    lid = f"{uidx}_{tidx}_{lesson_idx}"
                    ltitle = lesson["lessonTitle"]
                    lspec = lesson["spec_content"]
                    filtered_lessons.append({
                        "lesson_id": lid,
                        "lesson_title": ltitle,
                        "lesson_spec": lspec
                    })

        if not filtered_lessons:
            return  # Nothing to generate

        # 2. Shuffle the lessons to randomize session assignments
        random.shuffle(filtered_lessons)

        # 3. Partition into sessions of size <= max_lessons_per_session
        sessions: List[List[Dict[str, Any]]] = []
        for i in range(0, len(filtered_lessons), max_lessons_per_session):
            sessions.append(filtered_lessons[i : i + max_lessons_per_session])

        # 4. For each session, generate a LessonPerformance for each lesson
        for session_idx, lesson_batch in enumerate(sessions, start=1):
            for lesson_meta in lesson_batch:
                lid = lesson_meta["lesson_id"]
                ltitle = lesson_meta["lesson_title"]
                lspec = lesson_meta["lesson_spec"]

                # Adjust noise by motivation: more motivated → less noise
                if mot >= 0.7:
                    noise = base_noise * 0.5
                elif mot >= 0.3:
                    noise = base_noise
                else:
                    noise = base_noise * 1.5

                # Derive random mastery around prior knowledge
                mastery = clamp_val(prior + random.gauss(0, noise))
                # Confidence = mastery plus a bit of noise
                confidence = clamp_val(mastery + random.gauss(0, noise * 0.5))

                # Determine attempt/time/help bounds
                max_attempts = max(1, int((1 - prior) * 5))
                max_time = max(1, int(mastery * 60))
                max_helps = max(0, int((1 - mastery) * 3))

                # Random draws
                attempts = random.randint(1, max_attempts)
                time_spent = random.randint(1, max_time)
                completion = clamp_val(mastery + random.gauss(0, noise))
                retries = clamp_val((1 - mastery) * random.random())
                helps = random.randint(0, max_helps)
                hints = clamp_val(helps / 5.0)

                # Use session_idx as last_attempt
                last_session_index = session_idx

                lp = LessonPerformance(
                    lesson_id=lid,
                    lesson_title=ltitle,
                    lesson_spec=lspec,
                    mastery_score=mastery,
                    confidence_score=confidence,
                    completion_rate=completion,
                    total_attempts=attempts,
                    total_time_spent=time_spent,
                    last_attempt=last_session_index,
                    retry_frequency=retries,
                    help_requests=helps,
                    hint_usage_rate=hints,
                )

                # Store in profile
                profile.lesson_performances[lid] = lp

@dataclass
class TopicPerformance:
    """
    Aggregate performance metrics across multiple lessons within a topic.

    Attributes:
        topic_id: Identifier for the topic being aggregated.
        topic_title: Human-readable title of the topic.
        lessons: List of LessonPerformance instances for this topic.

        The following fields are automatically computed in __post_init__:
        num_lessons: Number of lessons in this topic.
        average_mastery: Mean of mastery_score across all lessons.
        average_confidence: Mean of confidence_score across all lessons.
        overall_completion_rate: Mean of completion_rate across all lessons.
        total_attempts: Sum of total_attempts across all lessons.
        total_time_spent: Sum of total_time_spent (in minutes) across all lessons.
        last_attempt_session: The most recent session index among lessons (max of last_attempt).
        average_retry_frequency: Mean of retry_frequency across lessons.
        total_help_requests: Sum of help_requests across all lessons.
        average_hint_usage_rate: Mean of hint_usage_rate across lessons.
        average_engagement_score: Mean of engagement scores computed from each lesson.
    """
    topic_id: str
    topic_title: str
    lessons: List["LessonPerformance"] = field(default_factory=list)

    # Aggregated metrics (populated in __post_init__)
    num_lessons: int = field(init=False)
    average_mastery: float = field(init=False)
    average_confidence: float = field(init=False)
    overall_completion_rate: float = field(init=False)
    total_attempts: int = field(init=False)
    total_time_spent: int = field(init=False)
    last_attempt_session: Optional[int] = field(init=False)
    average_retry_frequency: float = field(init=False)
    total_help_requests: int = field(init=False)
    average_hint_usage_rate: float = field(init=False)
    average_engagement_score: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Automatically calculate all aggregated metrics after initialization.
        """
        self.calculate_aggregates()

    def calculate_aggregates(self) -> None:
        """
        Compute and populate all aggregated metrics based on the `lessons` list.
        If there are no lessons, all aggregated numeric fields are set to zero,
        and last_attempt_session is set to None.
        """
        n = len(self.lessons)
        self.num_lessons = n

        if n == 0:
            # No lessons: set defaults
            self.average_mastery = 0.0
            self.average_confidence = 0.0
            self.overall_completion_rate = 0.0
            self.total_attempts = 0
            self.total_time_spent = 0
            self.last_attempt_session = None
            self.average_retry_frequency = 0.0
            self.total_help_requests = 0
            self.average_hint_usage_rate = 0.0
            self.average_engagement_score = 0.0
            return

        # Sum up raw values
        mastery_sum = 0.0
        confidence_sum = 0.0
        completion_sum = 0.0
        attempts_sum = 0
        time_spent_sum = 0
        retry_sum = 0.0
        helps_sum = 0
        hints_sum = 0.0
        engagement_sum = 0.0

        # Track most recent session index
        latest_session: Optional[int] = None

        for lesson in self.lessons:
            mastery_sum += lesson.mastery_score
            confidence_sum += lesson.confidence_score
            completion_sum += lesson.completion_rate
            attempts_sum += lesson.total_attempts
            time_spent_sum += lesson.total_time_spent
            retry_sum += lesson.retry_frequency
            helps_sum += lesson.help_requests
            hints_sum += lesson.hint_usage_rate

            # Compute engagement for this lesson
            engagement_sum += lesson.get_engagement_score()

            # Determine the most recent last_attempt (session index)
            if lesson.last_attempt is not None:
                if (latest_session is None) or (lesson.last_attempt > latest_session):
                    latest_session = lesson.last_attempt

        # Populate aggregated fields
        self.average_mastery = mastery_sum / n
        self.average_confidence = confidence_sum / n
        self.overall_completion_rate = completion_sum / n
        self.total_attempts = attempts_sum
        self.total_time_spent = time_spent_sum
        self.last_attempt_session = latest_session
        self.average_retry_frequency = retry_sum / n
        self.total_help_requests = helps_sum
        self.average_hint_usage_rate = hints_sum / n
        self.average_engagement_score = engagement_sum / n

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize this TopicPerformance instance (aggregated metrics only)
        to a dictionary suitable for JSON encoding.

        Returns:
            A dict containing:
              - "topic_id"
              - "topic_title"
              - Aggregated numeric fields (num_lessons, average_mastery, etc.)
        """
        return {
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "num_lessons": self.num_lessons,
            "average_mastery": self.average_mastery,
            "average_confidence": self.average_confidence,
            "overall_completion_rate": self.overall_completion_rate,
            "total_attempts": self.total_attempts,
            "total_time_spent": self.total_time_spent,
            "last_attempt_session": self.last_attempt_session,
            "average_retry_frequency": self.average_retry_frequency,
            "total_help_requests": self.total_help_requests,
            "average_hint_usage_rate": self.average_hint_usage_rate,
            "average_engagement_score": self.average_engagement_score,
        }

    @staticmethod
    def from_json_dict(data: Dict[str, Any]) -> "TopicPerformance":
        """
        Reconstruct a TopicPerformance instance from a JSON-like dictionary
        containing aggregated metrics. Since lesson-level details are not present,
        the `lessons` list will be initialized empty, and aggregated fields set
        directly from the provided data.

        Args:
            data: A dict containing keys matching those from to_json_dict().

        Returns:
            A TopicPerformance object with aggregated metrics populated.
        """
        tp = TopicPerformance(
            topic_id=data["topic_id"],
            topic_title=data.get("topic_title", ""),
            lessons=[]
        )

        # Assign aggregated fields directly
        tp.num_lessons = data.get("num_lessons", 0)
        tp.average_mastery = data.get("average_mastery", 0.0)
        tp.average_confidence = data.get("average_confidence", 0.0)
        tp.overall_completion_rate = data.get("overall_completion_rate", 0.0)
        tp.total_attempts = data.get("total_attempts", 0)
        tp.total_time_spent = data.get("total_time_spent", 0)
        tp.last_attempt_session = data.get("last_attempt_session", None)
        tp.average_retry_frequency = data.get("average_retry_frequency", 0.0)
        tp.total_help_requests = data.get("total_help_requests", 0)
        tp.average_hint_usage_rate = data.get("average_hint_usage_rate", 0.0)
        tp.average_engagement_score = data.get("average_engagement_score", 0.0)

        return tp


@dataclass
class UnitPerformance:
    """
    Aggregate performance metrics across multiple topics within a unit.

    Attributes:
        unit_id: Identifier for the unit being aggregated (e.g., "1").
        unit_title: Human-readable title of the unit.
        topics: List of TopicPerformance instances belonging to this unit.

        The following fields are automatically computed in __post_init__:
        num_topics: Number of topics in this unit.
        average_mastery: Mean of average_mastery across all topics.
        average_confidence: Mean of average_confidence across all topics.
        overall_completion_rate: Mean of overall_completion_rate across all topics.
        total_attempts: Sum of total_attempts across all topics.
        total_time_spent: Sum of total_time_spent across all topics.
        last_attempt_session: The most recent session index among topics.
        average_retry_frequency: Mean of average_retry_frequency across topics.
        total_help_requests: Sum of total_help_requests across all topics.
        average_hint_usage_rate: Mean of average_hint_usage_rate across topics.
        average_engagement_score: Mean of average_engagement_score across topics.
    """
    unit_id: str
    unit_title: str
    topics: List["TopicPerformance"] = field(default_factory=list)

    # Aggregated metrics (populated in __post_init__)
    num_topics: int = field(init=False)
    average_mastery: float = field(init=False)
    average_confidence: float = field(init=False)
    overall_completion_rate: float = field(init=False)
    total_attempts: int = field(init=False)
    total_time_spent: int = field(init=False)
    last_attempt_session: Optional[int] = field(init=False)
    average_retry_frequency: float = field(init=False)
    total_help_requests: int = field(init=False)
    average_hint_usage_rate: float = field(init=False)
    average_engagement_score: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Automatically calculate all aggregated metrics after initialization.
        """
        self.calculate_aggregates()

    def calculate_aggregates(self) -> None:
        """
        Compute and populate all aggregated metrics based on the `topics` list.
        If there are no topics, all aggregated numeric fields are set to zero,
        and last_attempt_session is set to None.
        """
        n = len(self.topics)
        self.num_topics = n

        if n == 0:
            # No topics: set defaults
            self.average_mastery = 0.0
            self.average_confidence = 0.0
            self.overall_completion_rate = 0.0
            self.total_attempts = 0
            self.total_time_spent = 0
            self.last_attempt_session = None
            self.average_retry_frequency = 0.0
            self.total_help_requests = 0
            self.average_hint_usage_rate = 0.0
            self.average_engagement_score = 0.0
            return

        # Sum up aggregated values from each topic
        mastery_sum = 0.0
        confidence_sum = 0.0
        completion_sum = 0.0
        attempts_sum = 0
        time_spent_sum = 0
        retry_sum = 0.0
        helps_sum = 0
        hints_sum = 0.0
        engagement_sum = 0.0

        # Track most recent session index among topics
        latest_session: Optional[int] = None

        for topic in self.topics:
            mastery_sum += topic.average_mastery
            confidence_sum += topic.average_confidence
            completion_sum += topic.overall_completion_rate
            attempts_sum += topic.total_attempts
            time_spent_sum += topic.total_time_spent
            retry_sum += topic.average_retry_frequency
            helps_sum += topic.total_help_requests
            hints_sum += topic.average_hint_usage_rate
            engagement_sum += topic.average_engagement_score

            if topic.last_attempt_session is not None:
                if (latest_session is None) or (topic.last_attempt_session > latest_session):
                    latest_session = topic.last_attempt_session

        # Populate aggregated fields
        self.average_mastery = mastery_sum / n
        self.average_confidence = confidence_sum / n
        self.overall_completion_rate = completion_sum / n
        self.total_attempts = attempts_sum
        self.total_time_spent = time_spent_sum
        self.last_attempt_session = latest_session
        self.average_retry_frequency = retry_sum / n
        self.total_help_requests = helps_sum
        self.average_hint_usage_rate = hints_sum / n
        self.average_engagement_score = engagement_sum / n

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize this UnitPerformance instance (aggregated metrics only)
        to a dictionary suitable for JSON encoding.

        Returns:
            A dict containing:
              - "unit_id"
              - "unit_title"
              - Aggregated numeric fields (num_topics, average_mastery, etc.)
        """
        return {
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "num_topics": self.num_topics,
            "average_mastery": self.average_mastery,
            "average_confidence": self.average_confidence,
            "overall_completion_rate": self.overall_completion_rate,
            "total_attempts": self.total_attempts,
            "total_time_spent": self.total_time_spent,
            "last_attempt_session": self.last_attempt_session,
            "average_retry_frequency": self.average_retry_frequency,
            "total_help_requests": self.total_help_requests,
            "average_hint_usage_rate": self.average_hint_usage_rate,
            "average_engagement_score": self.average_engagement_score,
        }

    @staticmethod
    def from_json_dict(data: Dict[str, Any]) -> "UnitPerformance":
        """
        Reconstruct a UnitPerformance instance from a JSON-like dictionary
        containing aggregated metrics. Since topic-level details are not present,
        the `topics` list will be initialized empty, and aggregated fields set
        directly from the provided data.

        Args:
            data: A dict containing keys matching those from to_json_dict().

        Returns:
            A UnitPerformance object with aggregated metrics populated.
        """
        up = UnitPerformance(
            unit_id=data["unit_id"],
            unit_title=data.get("unit_title", ""),
            topics=[]
        )

        # Assign aggregated fields directly
        up.num_topics = data.get("num_topics", 0)
        up.average_mastery = data.get("average_mastery", 0.0)
        up.average_confidence = data.get("average_confidence", 0.0)
        up.overall_completion_rate = data.get("overall_completion_rate", 0.0)
        up.total_attempts = data.get("total_attempts", 0)
        up.total_time_spent = data.get("total_time_spent", 0)
        up.last_attempt_session = data.get("last_attempt_session", None)
        up.average_retry_frequency = data.get("average_retry_frequency", 0.0)
        up.total_help_requests = data.get("total_help_requests", 0)
        up.average_hint_usage_rate = data.get("average_hint_usage_rate", 0.0)
        up.average_engagement_score = data.get("average_engagement_score", 0.0)

        return up

        

class Profile:
    """Simple container for holding lesson_performances."""
    def __init__(self):
        self.lesson_performances: Dict[str, LessonPerformance] = {}
        # self.topic_performances: Dict[str, TopicPerformance] = {}
        # self.unit_performances: Dict[str, UnitPerformance] = {}
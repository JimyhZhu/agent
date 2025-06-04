from typing import List, Dict, Optional, Set, Tuple, Any, Union
from enum import Enum
import random
from models import LessonPerformance, TopicPerformance, UnitPerformance, clamp_val, Profile
import os
import json

class LearningEnvironment:
    def __init__(self, profile: Any, curriculum: List[Dict[str, Any]], session_idx: int = 0):
        self.profile = profile
        self.curriculum = curriculum
        

    def step(self, info: Dict[str, Any]) -> Tuple[List["LessonPerformance"], List["TopicPerformance"], List["UnitPerformance"]]:
        """
        Simulate one “session” of learning. `info` is expected to contain:
        - "selected_lessons": List[str]
        - "mode_map": Dict[str, "teach" or "assess"]
        - (optionally) other fields like "lesson_recommendation_reason", etc.

        What this does:
        1. Increment the session index.
        2. For each lesson in selected_lessons:
            - Fetch its LessonPerformance from self.profile.
            - Update that LessonPerformance’s mastery & confidence according to mode.
            - Set its last_attempt = new_session_idx.
            - Slightly bump retry/hint/other metrics as a by‐product of “interaction.”
        3. After updating all chosen lessons, rebuild TopicPerformance and UnitPerformance
            objects for any topic/unit that contains one of these lessons.
        4. Return three lists: [updated LessonPerformance objects], [TopicPerformance], [UnitPerformance].
        """
        ############################################################
        # 1) Bump the session index
        ############################################################
        session_idx =  info.get("session_idx", 0) +1
        print(f"[DEBUG] Starting session {session_idx}")

        ############################################################
        # 2) Update each selected lesson’s performance
        ############################################################
        updated_lesson_perfs: List[LessonPerformance] = []
        selected_lessons = info.get("selected_lessons", [])
        mode_map: Dict[str, str] = info.get("mode_map", {})

        print(f"[DEBUG] Selected lessons for this session: {selected_lessons}")
        print(f"[DEBUG] Mode map: {mode_map}")

        for lesson_id in selected_lessons:
            print(f"\n[DEBUG] Processing lesson {lesson_id}")
            lp: LessonPerformance = self.profile.lesson_performances.get(lesson_id)
            if lp is None:
                print(f"[DEBUG] Lesson {lesson_id} not found in profile, creating new LessonPerformance")
                lesson_meta = self._find_lesson_meta(lesson_id)
                if lesson_meta is None:
                    print(f"[WARNING] Couldn't find metadata for {lesson_id}, skipping.")
                    continue
                lp = LessonPerformance(
                    lesson_id=lesson_id,
                    lesson_title=lesson_meta["lessonTitle"],
                    lesson_spec=lesson_meta["lessonSpec"],
                    mastery_score=0.0,
                    confidence_score=0.0,
                    completion_rate=0.0,
                    total_attempts=0,
                    total_time_spent=0,
                    last_attempt=None,
                    retry_frequency=0.0,
                    help_requests=0,
                    hint_usage_rate=0.0,
                )
                self.profile.lesson_performances[lesson_id] = lp

            # 2.a) Increment total_attempts & total_time_spent (simulate some interaction)
            lp.total_attempts += 1
            time_spent = random.randint(5, 20)  # e.g. user spends 5–20 minutes on each lesson
            lp.total_time_spent += time_spent
            print(f"[DEBUG]  - total_attempts -> {lp.total_attempts}, added time_spent={time_spent}, total_time_spent -> {lp.total_time_spent}")

            # 2.b) Decide how much to bump mastery/confidence:
            mode = mode_map.get(lesson_id, "assess")
            old_mastery = lp.mastery_score
            old_conf = lp.confidence_score
            print(f"[DEBUG]  - old_mastery={old_mastery:.3f}, old_confidence={old_conf:.3f}, mode={mode}")

            if mode == "teach":
                # Introduce or re‐teach: bigger gain in mastery, smaller gain in confidence
                mastery_gain = (1.0 - old_mastery) * 0.15
                confidence_gain = (1.0 - old_conf) * 0.10
            else:  # mode == "assess"
                # Reinforce: smaller mastery gain, bigger confidence gain
                mastery_gain = (1.0 - old_mastery) * 0.05
                confidence_gain = (1.0 - old_conf) * 0.20

            # 2.c) Add a bit of random noise to simulate variability
            mastery_gain *= random.uniform(0.8, 1.2)
            confidence_gain *= random.uniform(0.8, 1.2)
            print(f"[DEBUG]  - raw mastery_gain={mastery_gain:.3f}, raw confidence_gain={confidence_gain:.3f}")

            # 2.d) Update the LessonPerformance
            new_mastery = clamp_val(old_mastery + mastery_gain)
            new_conf = clamp_val(old_conf + confidence_gain)
            lp.mastery_score = new_mastery
            lp.confidence_score = new_conf
            print(f"[DEBUG]  - new_mastery={new_mastery:.3f}, new_confidence={new_conf:.3f}")

            # 2.e) Simulate help/hints usage occasionally
            if random.random() < 0.10:
                lp.help_requests += 1
                lp.hint_usage_rate = clamp_val(lp.hint_usage_rate + 0.05)
                print(f"[DEBUG]  - help request simulated: help_requests={lp.help_requests}, hint_usage_rate={lp.hint_usage_rate:.3f}")

            # 2.f) Update retry_frequency as fraction of attempts where mastery was low
            if old_mastery < 0.5:
                lp.retry_frequency = clamp_val(lp.retry_frequency + 0.1)
                print(f"[DEBUG]  - old_mastery < 0.5, retry_frequency bumped to {lp.retry_frequency:.3f}")

            # 2.g) Update completion_rate if mastery is above threshold
            if new_mastery > 0.8:
                lp.completion_rate = clamp_val(lp.completion_rate + 0.2)
                print(f"[DEBUG]  - new_mastery > 0.8, completion_rate bumped to {lp.completion_rate:.3f}")

            # 2.h) Set last_attempt to current session index
            lp.last_attempt = session_idx
            print(f"[DEBUG]  - last_attempt set to session {session_idx}")

            updated_lesson_perfs.append(lp)

        ############################################################
        # 3) Re‐compute topic‐level performance for any affected topics
        ############################################################
        affected_topics: Set[str] = set()
        for lesson_id in selected_lessons:
            parts = lesson_id.split("_")
            if len(parts) >= 2:
                topic_id = f"{parts[0]}_{parts[1]}"
                affected_topics.add(topic_id)

        print(f"\n[DEBUG] Affected topics: {sorted(affected_topics)}")
        updated_topic_perfs: List[TopicPerformance] = []
        for t_id in affected_topics:
            lesson_list: List[LessonPerformance] = [
                lp for lid, lp in self.profile.lesson_performances.items() if lid.startswith(t_id + "_")
            ]
            print(f"[DEBUG]  - Rebuilding TopicPerformance for {t_id}, {len(lesson_list)} lessons found")
            if lesson_list:
                tp = TopicPerformance(topic_id=t_id, topic_title=self._lookup_topic_title(t_id), lessons=lesson_list)
                updated_topic_perfs.append(tp)
                print(f"[DEBUG]    • {t_id}: avg_mastery={tp.average_mastery:.3f}, avg_confidence={tp.average_confidence:.3f}")

        ############################################################
        # 4) Re‐compute unit‐level performance for any affected units
        ############################################################
        affected_units: Set[str] = set()
        for t_id in affected_topics:
            unit_id = t_id.split("_")[0]
            affected_units.add(unit_id)

        print(f"\n[DEBUG] Affected units: {sorted(affected_units)}")
        updated_unit_perfs: List[UnitPerformance] = []
        for u_id in affected_units:
            # Gather all TopicPerformance objects for this unit
            topic_objs: List[TopicPerformance] = [
                tp for tp in updated_topic_perfs if tp.topic_id.startswith(u_id + "_")
            ]

            # Also rebuild any other topics in this unit that weren’t updated this session
            for unit in self.curriculum:
                if str(unit["unitIndex"]) == u_id:
                    for topic in unit["topics"]:
                        tid = f"{u_id}_{topic['topicIndex']}"
                        if any(tp.topic_id == tid for tp in topic_objs):
                            continue
                        lesson_list = [
                            lp for lid, lp in self.profile.lesson_performances.items() if lid.startswith(tid + "_")
                        ]
                        if lesson_list:
                            tp = TopicPerformance(topic_id=tid, topic_title=topic["topicTitle"], lessons=lesson_list)
                            topic_objs.append(tp)
                            print(f"[DEBUG]  - Also rebuilt TopicPerformance for {tid}: avg_mastery={tp.average_mastery:.3f}, avg_confidence={tp.average_confidence:.3f}")
                    break

            if topic_objs:
                unit_title = self._lookup_unit_title(u_id)
                up = UnitPerformance(unit_id=u_id, unit_title=unit_title, topics=topic_objs)
                updated_unit_perfs.append(up)
                print(f"[DEBUG]  - Built UnitPerformance for {u_id}: avg_mastery={up.average_mastery:.3f}, avg_confidence={up.average_confidence:.3f}")

        ############################################################
        # 5) Return everything
        ############################################################
        print(f"\n[DEBUG] Session {session_idx} complete. Returning updated performances.\n")
        return updated_lesson_perfs, updated_topic_perfs, updated_unit_perfs


    def _find_lesson_meta(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up the lesson’s metadata (title, spec) in self.curriculum.
        Returns { "lessonTitle": ..., "lessonSpec": ... } or None if not found.
        """
        parts = lesson_id.split("_")
        if len(parts) != 3:
            return None
        uidx, tidx, lidx = parts
        for unit in self.curriculum:
            if str(unit["unitIndex"]) == uidx:
                for topic in unit["topics"]:
                    if str(topic["topicIndex"]) == tidx:
                        for lesson in topic["lessons"]:
                            if str(lesson["lessonIndex"]) == lidx:
                                return {
                                    "lessonTitle": lesson["lessonTitle"],
                                    "lessonSpec": lesson.get("lessonSpec", "")
                                }
        return None

    def _lookup_topic_title(self, topic_id: str) -> str:
        """
        Given "unit_topic" (e.g. "1_2"), return the topicTitle from curriculum.
        """
        parts = topic_id.split("_")
        if len(parts) != 2:
            return ""
        uidx, tidx = parts
        for unit in self.curriculum:
            if str(unit["unitIndex"]) == uidx:
                for topic in unit["topics"]:
                    if str(topic["topicIndex"]) == tidx:
                        return topic["topicTitle"]
        return ""

    def _lookup_unit_title(self, unit_id: str) -> str:
        """
        Given "unit" (e.g. "1"), return the unitTitle from curriculum.
        """
        for unit in self.curriculum:
            if str(unit["unitIndex"]) == unit_id:
                return unit["unitTitle"]
        return ""
    
    def generate_random_demo_data(
        self,
        student_characteristics: Dict[str, float],
        *,
        max_unit_index: Optional[int] = None,
        max_topic_index: Optional[int] = None,
        max_lessons_per_session: int = 5,
        max_sessions: Optional[int] = None,
        topic_baseline: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Populate `self.profile.lesson_performances` with demo data, but with optional
        per-topic “baseline” mastery & confidence levels.

        Args:
            student_characteristics: Dict with keys:
                - "motivation": float in [0.0, 1.0]
                - "prior_knowledge": float in [0.0, 1.0]
                - "attention_span": float in [0.0, 1.0]
            max_unit_index: Only include units with unitIndex ≤ this (if provided).
            max_topic_index: Only include topics with topicIndex ≤ this (if provided).
            max_lessons_per_session: How many lessons per simulated session.
            max_sessions: If provided, cap total number of sessions at this.
            topic_baseline: Optional dict mapping each topic_id (e.g. "1_2") to a dict:
                {
                  "mastery": float in [0,1],
                  "confidence": float in [0,1]
                }
                If a topic_id is missing, fallback to the global `prior_knowledge`.

        Raises:
            ValueError if `student_characteristics` is missing any of the required keys.
        """
        # 1. Validate student_characteristics
        required_keys = ["motivation", "prior_knowledge", "attention_span"]
        if not all(k in student_characteristics for k in required_keys):
            raise ValueError(f"Need {required_keys} in student_characteristics")

        mot = clamp_val(student_characteristics["motivation"])
        global_prior = clamp_val(student_characteristics["prior_knowledge"])
        att = clamp_val(student_characteristics["attention_span"])
        base_noise = (1.0 - att) * 0.3

        # 2. Build a flat list of all (lesson_id, lesson_title, lesson_spec, topic_id) that meet unit/topic thresholds
        filtered_lessons = []
        for unit in self.curriculum:
            uidx = unit.get("unitIndex", 0)
            if (max_unit_index is not None) and (uidx > max_unit_index):
                continue
            for topic in unit.get("topics", []):
                tidx = topic.get("topicIndex", 0)
                if (max_topic_index is not None) and (tidx > max_topic_index):
                    continue

                topic_id = f"{uidx}_{tidx}"
                for lesson in topic.get("lessons", []):
                    lidx = lesson["lessonIndex"]
                    lid = f"{uidx}_{tidx}_{lidx}"
                    ltitle = lesson["lessonTitle"]
                    lspec = lesson.get("spec_content", "")

                    filtered_lessons.append({
                        "lesson_id": lid,
                        "lesson_title": ltitle,
                        "lesson_spec": lspec,
                        "topic_id": topic_id
                    })

        if not filtered_lessons:
            return  # Nothing to generate

        # 3. Shuffle the lessons to randomize session assignment
        random.shuffle(filtered_lessons)

        # 4. Partition into sessions of size ≤ max_lessons_per_session
        all_partitions: List[List[Dict[str, Any]]] = []
        for i in range(0, len(filtered_lessons), max_lessons_per_session):
            all_partitions.append(filtered_lessons[i : i + max_lessons_per_session])

        # 5. If max_sessions is set, truncate the partitions
        if max_sessions is not None:
            all_partitions = all_partitions[:max_sessions]

        # 6. Generate a LessonPerformance for each lesson in each partition
        for session_idx, lesson_batch in enumerate(all_partitions, start=1):
            for lesson_meta in lesson_batch:
                lid = lesson_meta["lesson_id"]
                ltitle = lesson_meta["lesson_title"]
                lspec = lesson_meta["lesson_spec"]
                topic_id = lesson_meta["topic_id"]

                # a) Determine this lesson’s baseline mastery & confidence:
                if topic_baseline and (topic_id in topic_baseline):
                    baseline_mastery = clamp_val(topic_baseline[topic_id].get("mastery", global_prior))
                    baseline_conf = clamp_val(topic_baseline[topic_id].get("confidence", baseline_mastery))
                else:
                    baseline_mastery = global_prior
                    baseline_conf = global_prior

                # b) Adjust noise by motivation: more motivated → less noise
                if mot >= 0.7:
                    noise = base_noise * 0.5
                elif mot >= 0.3:
                    noise = base_noise
                else:
                    noise = base_noise * 1.5

                # c) Derive random mastery around baseline_mastery
                mastery = clamp_val(baseline_mastery + random.gauss(0, noise))
                # d) Derive random confidence around baseline_conf (+ smaller noise)
                confidence = clamp_val(baseline_conf + random.gauss(0, noise * 0.5))

                # e) Determine attempt/time/help bounds
                max_attempts = max(1, int((1 - baseline_mastery) * 5))
                max_time = max(1, int(mastery * 60))
                max_helps = max(0, int((1 - mastery) * 3))

                # f) Random draws
                attempts = random.randint(1, max_attempts)
                time_spent = random.randint(1, max_time)
                completion = clamp_val(mastery + random.gauss(0, noise))
                retries = clamp_val((1 - mastery) * random.random())
                helps = random.randint(0, max_helps)
                hints = clamp_val(helps / 5.0)

                # g) Package into a LessonPerformance
                lp = LessonPerformance(
                    lesson_id=lid,
                    lesson_title=ltitle,
                    lesson_spec=lspec,
                    mastery_score=mastery,
                    confidence_score=confidence,
                    completion_rate=completion,
                    total_attempts=attempts,
                    total_time_spent=time_spent,
                    last_attempt=session_idx,
                    retry_frequency=retries,
                    help_requests=helps,
                    hint_usage_rate=hints,
                )

                # h) Store or overwrite in profile
                self.profile.lesson_performances[lid] = lp
        



if __name__ == "__main__":

    MEMORY_FILE = "test_memory.json"
    if os.path.isfile(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    # --- 1. Define a small sample curriculum with units, topics, and lessons ---
    curriculum = json.load(open("/Users/jimzhu/work_dir/medly/math_curriculum.json"))

    # --- 2. Generate random LessonPerformance data for a mock profile ---
    profile = Profile()
    env = LearningEnvironment(profile, curriculum)
    # 2) Define per-topic baselines:
    topic_baseline = {
        "1_0": {"mastery": 0.2, "confidence": 0.3},
        "1_1": {"mastery": 0.8, "confidence": 0.7},
        "2_0": {"mastery": 0.5, "confidence": 0.5},
        # etc…
    }

    # 3) Simulate demo data, up to 2 units, 2 topics, 3 lessons per session, 2 sessions:
    env.generate_random_demo_data(
        student_characteristics={
            "motivation": 0.6,
            "prior_knowledge": 0.4,
            "attention_span": 0.7
        },
        max_unit_index=2,
        max_topic_index=2,
        max_lessons_per_session=3,
        max_sessions=2,
        topic_baseline=topic_baseline
    )

    # 4) Inspect a few generated LessonPerformance records
    for lid, lp in sorted(profile.lesson_performances.items(), key=lambda x: x[1].last_attempt):
        print(f"{lid} → last_attempt={lp.last_attempt}, total_attempts={lp.total_attempts}, "
          f"mastery={lp.mastery_score:.3f}, confidence={lp.confidence_score:.3f}")

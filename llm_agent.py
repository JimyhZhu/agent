import os
import json
import random
import asyncio
import logging
import openai
import tiktoken

from typing import List, Dict, Optional, Any
from models import Profile, LessonPerformance, TopicPerformance, UnitPerformance
from memory_manager import MemoryManager
from learning_environment import LearningEnvironment

# ─────────── Configure logger ───────────
logger = logging.getLogger("LLMAgent")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
console_handler.setFormatter(console_formatter)

# Markdown file handler
log_md_path = "llm_agent_debug_log.md"
file_handler = logging.FileHandler(log_md_path, mode='w')  # 'w' to overwrite
file_formatter = logging.Formatter("**[%(levelname)s]** `%(asctime)s` - %(message)s")
file_handler.setFormatter(file_formatter)

# Add both handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    num_tokens = 0

    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(enc.encode(value))

    num_tokens += 2
    return num_tokens


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        try:
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
        
    async def generate(self, prompt: Any, max_tokens: Optional[int] = None) -> str:
        if isinstance(prompt, tuple):
            system_msg, user_msg = prompt
        else:
            system_msg = "You are an expert learning advisor."
            user_msg = prompt

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ]

        logger.debug("Total tokens in prompt: %d", count_tokens(messages))
        #logger.debug("System message:\n%s", json.dumps(system_msg, indent=2))
        #logger.debug("User payload / prompt content:\n%s", json.dumps(user_msg, indent=2))

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or 4000,
                temperature=0.7
            )
            raw = response.choices[0].message.content.strip()
            #logger.debug("Raw LLM response:\n%s", raw)
            return raw
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class LLMAgent:
    """
    An agent that leverages an LLM (via LLMClient) to propose, in two stages,
    first a set of topics, then lessons within those topics, for the next session.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = LLMClient(api_key=api_key, model=model)

    async def propose_next_topics(
        self,
        profile: Any,
        memory: Any,
        curriculum: List[Dict[str, Any]],
        interests: List[str],
        num_topics: int = 3,
        history_length: int = 3
    ) -> Dict[str, Any]:
        """
        Step 1: Ask the LLM to pick the top `num_topics` topics for the next session,
        based on topic‐level stats, previous session shifts, and interests.
        Returns a dict with "selected_topics" and "topic_recommendation_reason".
        """
        # a) Gather previous‐session topic data:
        all_sessions = memory.sessions or []
        recent_sessions = all_sessions[-history_length:]
        
        previous_sessions_payload: Dict[int, Dict[str, Any]] = {}
        for sess in recent_sessions:
            fragment: Dict[str, Any] = {
                "topic_recommendation_reason": sess.get("topic_recommendation_reason", {}),
                "lesson_recommendation_reason": sess.get("lesson_recommendation_reason", {}),
                "mastery_shift": sess.get("mastery_shift", 0.0),
                "confidence_shift": sess.get("confidence_shift", 0.0)
            }
            previous_sessions_payload[sess["session_id"]] = fragment

        # b) Build a list of topic‐level history and metadata:
        topic_history = []
        for unit in curriculum:
            uidx = str(unit["unitIndex"])
            for topic in unit["topics"]:
                tidx = str(topic["topicIndex"])
                topic_id = f"{uidx}_{tidx}"
                tp = memory.get_latest_topic_performance(topic_id)
                if tp is None:
                    topic_history.append({
                        "topic_id": topic_id,
                        "topic_title": topic["topicTitle"],
                        "last_attempt_session": None,
                        "average_mastery": 0.0,
                        "average_confidence": 0.0
                    })
                else:
                    tp_json = tp.to_json_dict()
                    topic_history.append({
                        "topic_id": topic_id,
                        "topic_title": tp_json["topic_title"],
                        "last_attempt_session": tp_json["last_attempt_session"],
                        "average_mastery": tp_json["average_mastery"],
                        "average_confidence": tp_json["average_confidence"]
                    })

        # c) Build a small prompt for topic selection:
        system_message = (
            "You are an expert learning advisor. Given a learner's recent topic-level "
            f"performance history (up to {history_length} past sessions), their interests, "
            "and the curriculum structure, select the top "
            f"{num_topics} topics for the next session."
        )

        user_payload = {
            "num_topics_required": num_topics,
            "interests": interests,
            "previous_sessions": previous_sessions_payload,
            "topic_data": topic_history,
            "instructions": [
                "1. Exclude any topic whose average_mastery ≥ 0.9 (consider it mastered).",
                "2. For each remaining topic, compute a priority score using:",
                "     • the gap between target mastery (1.0) and average_mastery,",
                "     • the gap between target confidence (1.0) and average_confidence,",
                "     • recency (last_attempt_session),",
                "     • whether the topic appeared (and with what reason) in previous sessions,",
                "     • and the learner's stated interests.",
                "3. Pick the top num_topics_required topics by descending priority.",
                "4. Output exactly a JSON object with keys:",
                "     - selected_topics: [topic_id, …],",
                "     - topic_recommendation_reason: { topic_id: reason_string, … }",
                "5. Do not wrap the JSON in markdown or extra commentary."
            ]
        }
        logger.info(f"propose_next_topics: system_message={system_message}")
        logger.info(f"propose_next_topics: user_payload={json.dumps(user_payload, indent=2)}")
        prompt = (system_message, json.dumps(user_payload))
        raw_response = await self.client.generate(prompt)

        # Parse LLM JSON response
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            parsed = json.loads(raw_response[start:end])

        # Debug: log parsed result
        logger.debug("[propose_next_topics] parsed LLM output:\n%s", json.dumps(parsed, indent=2))

        return {
            "selected_topics": parsed.get("selected_topics", []),
            "topic_recommendation_reason": parsed.get("topic_recommendation_reason", {})
        }

    async def propose_lessons_within_topics(
        self,
        profile: Any,
        memory: Any,
        curriculum: List[Dict[str, Any]],
        selected_topics: List[str],
        topic_recommendation_reason: Dict[str, str],
        previous_session: Dict[str, Any],
        interests: List[str],
        N: int = 4
    ) -> Dict[str, Any]:
        """
        Step 2: Given a shortlist of `selected_topics`, ask the LLM to select
        the top N lessons from within those topics. Returns a dict:
        {
            "selected_lessons": [...],
            "lesson_recommendation_reason": {...},
            "mode_map": {...},
            "reasoning_chain": ...
        }
        """
        # a) Collect lesson metadata for selected topics
        lesson_history = []
        topic_titles = {}
        
        for unit in curriculum:
            uidx = str(unit["unitIndex"])
            for topic in unit["topics"]:
                tidx = str(topic["topicIndex"])
                topic_id = f"{uidx}_{tidx}"
                if topic_id not in selected_topics:
                    continue
                topic_title = topic["topicTitle"]
                topic_titles[topic_id] = topic_title
                for lesson in topic["lessons"]:
                    lid = f"{topic_id}_{lesson['lessonIndex']}"
                    lp = memory.get_latest_lesson_performance(lid)
                    if lp is None:
                        lesson_history.append({
                            "lesson_id": lid,
                            "lesson_title": lesson["lessonTitle"],
                            "topic_id": topic_id,
                            "topic_title": topic_title,
                            "unit_id": uidx,
                            "mastery": 0.0,
                            "confidence": 0.0,
                            "retry": 0.0,
                            "last_attempt": None
                        })
                    else:
                        lesson_history.append({
                            "lesson_id": lid,
                            "lesson_title": lesson["lessonTitle"],
                            "topic_id": topic_id,
                            "topic_title": topic_title,
                            "unit_id": uidx,
                            "mastery": round(lp.mastery_score, 2),
                            "confidence": round(lp.confidence_score, 2),
                            "retry": round(lp.retry_frequency, 2),
                            "last_attempt": lp.last_attempt
                        })

        system_message = (
            "You are an intelligent lesson recommender.  "
            "You will receive a list of candidate lessons (with mastery, confidence, retry, recency, etc.) "
            "along with their parent topics and some history.  "
            "Your job is to:\n"
            "  1. First, classify each lesson as either 'teach' or 'assess' based on the rules below.\n"
            "  2. Then, compute a priority score for every lesson.\n"
            "  3. Finally, select exactly N lessons and return a clean JSON with mode and reasons.\n"
            "Follow the output format exactly and do not include any markdown or extra commentary."
        )

        user_payload = {
            "num_lessons_required": N,
            "selected_topics": [
                {
                    "topic_id": tid,
                    "topic_title": topic_titles.get(tid, ""),
                    "reason": topic_recommendation_reason.get(tid, "")
                }
                for tid in selected_topics
            ],
            "previous_session": previous_session,
            "lesson_history": lesson_history,
            "instructions": [
                # 1. CLASSIFICATION PHASE
                "1. For each lesson in 'lesson_history', determine its 'mode' (either 'teach' or 'assess') as follows:",
                "   • If last_attempt is null OR mastery < 0.5 OR lesson_id was in previous 'skipped_lessons', set mode = 'teach'.",
                "   • Otherwise, if confidence < 0.4, set mode = 'assess' (reinforce lower confidence).",
                "   • Otherwise, set mode = 'assess'.",
                "",
                # 2. SCORING PHASE
                "2. Compute a priority score for each lesson:",
                "   • mastery_gap = (1.0 - mastery), higher → higher priority.",
                "   • confidence_gap = (1.0 - confidence), higher → higher priority.",
                "   • retry_frequency (higher → higher priority).",
                "   • recency: if last_attempt is older (smaller session index or null) → boost priority.",
                "   • If lesson was in 'skipped_lessons' last session, add a small bonus.",
                "   • If lesson's parent topic is in 'selected_topics', add a topic relevance bonus.",
                "   • If lesson aligns with any 'interests', add a small bonus.",
                "",
                # 3. SELECTION PHASE
                "3. Combine all lessons into one list sorted by descending priority score. "
                "Pick the top num_lessons_required from that combined list.",
                "",
                # 4. OUTPUT FORMAT
                "4. Return exactly one JSON object with these keys (no extra text):",
                "   {",
                "     \"reasoning_chain\": { … },",
                "     \"selected_lessons\": [lesson_id, …],",
                "     \"lesson_recommendation_reason\": { lesson_id: reason_string, … },",
                "     \"mode_map\": { lesson_id: \"teach\" or \"assess\", … }",
                "   }",
                "",
                "5. Do NOT wrap the output in markdown or code fences—only raw JSON.",
                "6. Try to include at least one lesson from each 'selected_topics' if possible."
            ]
        }

     
        prompt = (system_message, json.dumps(user_payload))
        raw_response = await self.client.generate(prompt)

        
        # Parse LLM JSON response
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            parsed = json.loads(raw_response[start:end])


        return {
            "selected_lessons": parsed.get("selected_lessons", []),
            "lesson_recommendation_reason": parsed.get("lesson_recommendation_reason", {}),
            "mode_map": parsed.get("mode_map", {}),
            "reasoning_chain": parsed.get("reasoning_chain", {})
        }

    async def propose_next_session(
        self,
        profile: Any,
        memory: Any,
        curriculum: List[Dict[str, Any]],
        interests: List[str],
        num_topics: int = 3,
        num_lessons: int = 4
    ) -> Dict[str, Any]:
        """
        Orchestrates the two-step process:
          1. propose_next_topics → get a list of topic_ids
          2. propose_lessons_within_topics → get the final lesson recommendations
        """
        # 1. First, select topics
        proposed_topics_info = await self.propose_next_topics(
            profile=profile,
            memory=memory,
            curriculum=curriculum,
            interests=interests,
            num_topics=num_topics
        )

        selected_topics = proposed_topics_info["selected_topics"]
        topic_recommendation_reason = proposed_topics_info["topic_recommendation_reason"]

        
        # 2. Build the previous_session dict
        last_session = memory.sessions[-1] if memory.sessions else {}
        previous_session_info = {
            "selected_lessons": last_session.get("selected_lessons", []),
            "completed_lessons": last_session.get("completed_lessons", []),
            "skipped_lessons": last_session.get("skipped_lessons", []),
            "mastery_shift": last_session.get("mastery_shift", 0.0),
            "confidence_shift": last_session.get("confidence_shift", 0.0),
            "ai_support_used": last_session.get("ai_support_used", {}),
            "recommendation_reason": last_session.get("recommendation_reason", {})
        }

        # Debug: log previous session info
        #logger.debug("[propose_next_session] previous_session_info:\n%s", json.dumps(previous_session_info, indent=2))

        # 3. Then, pick lessons within those topics
        lessons_payload = await self.propose_lessons_within_topics(
            profile=profile,
            memory=memory,
            curriculum=curriculum,
            selected_topics=selected_topics,
            topic_recommendation_reason=topic_recommendation_reason,
            previous_session=previous_session_info,
            interests=interests,
            N=num_lessons
        )
        lessons_payload["selected_topics"] = selected_topics
        lessons_payload["topic_recommendation_reason"] = topic_recommendation_reason
        # Use length of sessions as next session index if session_idx is not present
        lessons_payload["session_idx"] = memory.sessions[-1].get("session_idx", len(memory.sessions)) + 1
        # Debug: log final lessons payload
        logger.debug("[propose_next_session] lessons_payload:\n%s", json.dumps(lessons_payload, indent=2))
        return lessons_payload


if __name__ == "__main__":
    MEMORY_FILE = "test_memory.json"
    if os.path.isfile(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    profile = Profile()
    curriculum = json.load(open("/Users/jimzhu/work_dir/medly/math_curriculum.json"))

    manager = MemoryManager(disk_path=MEMORY_FILE)
    env = LearningEnvironment(profile, curriculum)

    # Define per-topic baseline if desired
    topic_baseline = {
        "1_0": {"mastery": 0.2, "confidence": 0.3},
        "1_1": {"mastery": 0.8, "confidence": 0.7},
        "2_0": {"mastery": 0.5, "confidence": 0.5},
    }

    # Simulate demo data
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

    # Record those sessions back into memory
    manager.record_sessions_from_profile(profile, curriculum)

    # Reload memory
    manager_reload = MemoryManager(disk_path=MEMORY_FILE)

    llm_agent = LLMAgent(api_key=os.getenv("OPENAI_API_KEY"))

    # # Debug: propose topics
    # proposed_topics_info = asyncio.run(
    #     llm_agent.propose_next_topics(
    #         profile=profile,
    #         memory=manager_reload,
    #         curriculum=curriculum,
    #         interests=["Improve Algebra"]
    #     )
    # )
    # logger.info("=== Proposed Topics Info ===\n%s", json.dumps(proposed_topics_info, indent=2))

    # # Debug: propose lessons within a hypothetical set of topics
    # payload = asyncio.run(
    #     llm_agent.propose_lessons_within_topics(
    #         profile=profile,
    #         memory=manager_reload,
    #         curriculum=curriculum,
    #         selected_topics=["1_2", "1_0", "1_3"],
    #         topic_recommendation_reason=proposed_topics_info["topic_recommendation_reason"],
    #         previous_session={
    #             "selected_lessons": [],
    #             "completed_lessons": [],
    #             "skipped_lessons": [],
    #             "mastery_shift": 0.0,
    #             "confidence_shift": 0.0,
    #             "ai_support_used": {},
    #             "recommendation_reason": {}
    #         },
    #         interests=["Improve Algebra"],
    #         N=4
    #     )
    # )
    # logger.info("=== Proposed Lessons Payload ===\n%s", json.dumps(payload, indent=2))

    # Debug: propose the full next session
    info = asyncio.run(
        llm_agent.propose_next_session(
            profile=profile,
            memory=manager_reload,
            curriculum=curriculum,
            interests=["Improve Algebra"]
        )
    )
    logger.info("=== Proposed Next Session ===\n%s", json.dumps(info, indent=2))

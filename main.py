from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from collections import defaultdict
from enum import Enum
import os
import asyncio
import json

from models import LessonPerformance, TopicPerformance, UnitPerformance, clamp_val, Profile
from learning_environment import LearningEnvironment
from llm_agent import LLMAgent
from memory_manager import MemoryManager
import logging
from dotenv import load_dotenv
load_dotenv()

# ─────────── Configure logger ───────────
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s")
console_handler.setFormatter(console_formatter)

# Markdown file handler
log_md_path = "main_debug_log.md"
file_handler = logging.FileHandler(log_md_path, mode='w')  # 'w' to overwrite
file_formatter = logging.Formatter("**[%(levelname)s]** `%(asctime)s` - %(message)s")
file_handler.setFormatter(file_formatter)

# Add both handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

if __name__ == "__main__":
    # --- Setup: remove any existing test memory file so we start clean ---
    MEMORY_FILE = "test_memory.json"
    if os.path.isfile(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    # --- 1. Define a small sample curriculum with units, topics, and lessons ---
    curriculum = json.load(open("math_curriculum.json"))

    # --- 2. Generate random LessonPerformance data for a mock profile ---
    profile = Profile()

    student_characteristics = {
        "motivation": 0.6,
        "prior_knowledge": 0.4,
        "attention_span": 0.7,
    }

    topic_baseline = {
        "0_0": {"mastery": 0.2, "confidence": 0.3},
        "0_1": {"mastery": 0.8, "confidence": 0.7},
        "0_2": {"mastery": 0.5, "confidence": 0.5},
        "1_0": {"mastery": 0.2, "confidence": 0.3},
        "1_1": {"mastery": 0.8, "confidence": 0.7},
        "1_2": {"mastery": 0.5, "confidence": 0.5},
        "2_0": {"mastery": 0.5, "confidence": 0.5},
    }

    max_unit_index = 2
    max_topic_index = 5
    max_lessons_per_session = 5
    max_sessions = 10

    # This populates profile.lesson_performances with random data for every lessonId in curriculum
    env = LearningEnvironment(profile, curriculum)

    logger.info("Generating random demo data")
    env.generate_random_demo_data(student_characteristics,  max_unit_index=max_unit_index, max_topic_index=max_topic_index, max_lessons_per_session=max_lessons_per_session, max_sessions=max_sessions)
    
    
    logger.info(
    f"Random demo data generated with max_unit_index={max_unit_index}, "
    f"max_topic_index={max_topic_index}, max_lessons_per_session={max_lessons_per_session}, "
    f"max_sessions={max_sessions}, \n"
    f"student_characteristics={student_characteristics}, \n"
    f"topic_baseline={topic_baseline}"
    )

    # # --- 3. Instantiate MemoryManager and record one session ---
    manager = MemoryManager(disk_path=MEMORY_FILE)

    manager.record_sessions_from_profile(profile, curriculum)
    logger.info(f"After first record_session(), total sessions stored: {len(manager.sessions)}")
    #print(f"After first record_session(), total sessions stored: {len(manager.sessions)}")


    # --- 4. Simulate loading from disk by creating a new MemoryManager instance ---
    manager_reload = MemoryManager(disk_path=MEMORY_FILE)
    
    logger.info(f"After reload, total sessions stored: {len(manager_reload.sessions)}")

     
    llm_agent = LLMAgent(api_key=os.getenv("OPENAI_API_KEY"))
    info = asyncio.run(llm_agent.propose_next_session(profile=profile,
                                          memory=manager_reload,
                                           curriculum=curriculum, 
                                           interests=["Improve Algebra"]))
    
    logger.info("=== Proposed Next Session ===\n%s", json.dumps(info, indent=2))

    lesson_perfs, topic_perfs, unit_perfs = env.step(info)
    logger.info("Recording session")


    manager_reload.record_session(
        lesson_perfs=lesson_perfs,
        topic_perfs=topic_perfs,
        unit_perfs=unit_perfs,
        selected_lessons=info["selected_lessons"],
        lesson_recommendation_reason=info["lesson_recommendation_reason"],
        topic_recommendation_reason=info["topic_recommendation_reason"],
    )   

    logger.info(f"After recording session, total sessions stored: {len(manager_reload.sessions)}")
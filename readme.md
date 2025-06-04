# Adaptive Learning System

An intelligent learning platform leveraging LLM-powered recommendations to adapt dynamically to student performance and interests.

## Architecture

### Core Components

#### 1. **Learning Environment (`learning_environment.py`)**

* **Simulates learning sessions and tracks student performance:**

  * `step(info)`: Processes learning sessions and updates metrics.
  * `generate_random_demo_data()`: Creates customizable test data.


#### 2. **Memory Manager (`memory_manager.py`)**

* **Persistent JSON-based storage with caching:**

  * `_load_from_disk()` / `_save_to_disk()`: Manages session data persistence.
  * `record_session()` / `: Records sessions to memory.
  * `record_sessions_from_profile()`: Reconstruct sessions data based on previously generated perfromance data using the `generate_random_demo_data()` from learning_enviroment.py
 
* **Session-level performance tracking:**

  * Retrieval functions (`get_latest_lesson_performance`, `get_all_lesson_history`, etc.) maintain historical performance data.

#### 3. **LLM Agent (`llm_agent.py`)**

* **Intelligent recommendation engine leveraging OpenAI GPT:**

  * `propose_next_topics()`: Recommends topics considering history and interests.
  * `propose_lessons_within_topics()`: Selects and categorizes lessons (teach/assess) based on selected topics.
  * `propose_next_session()`: Generates complete session recommendations.

* **Recommendation factors:** Student performance, interests, curriculum structure, and learning patterns.

#### 4. **Data Models (`models.py`)**

* **Individual lesson tracking (`LessonPerformance`):**

  * Mastery, confidence, completion, attempts, engagement metrics.
* **Aggregated metrics:**

  * Topic-level (`TopicPerformance`): Averages mastery/confidence, tracks completion and attempts.
  * Unit-level (`UnitPerformance`): Aggregates topic performances and monitors progress.
* **Student profile:**

  * Stores and manages learner characteristics and lesson performance data.

## Decision-Making Flow

### Macro-to-Micro Process

#### **Topic Selection (Macro)**

* Analyzes performance trends, student interests, and identifies knowledge gaps.

#### **Lesson Selection (Micro)**

* Selects lessons within chosen topics based on performance, content coverage and student's interest.

### Mode Determination (Teach vs. Assess)

#### **Teach Mode Criteria:**

* Low mastery (<0.5), high retry frequency, recent topic introduction, identified knowledge gaps.

#### **Assess Mode Criteria:**

* Moderate/high mastery scores, confidence-building needs, previous sessions completed.

### Performance Updates

* **Teach Mode:** Larger mastery gains, smaller confidence gains.
* **Assess Mode:** Smaller mastery gains, larger confidence gains.

## Data Flow

1. Load student characteristics and curriculum.
2. Generate initial performance data.
3. Record session history.
4. LLM analysis and recommendation for next session.
5. Simulate session and update performance data.

## Memory Implementation

### Storage Format

* Persistent JSON with session details:

  * Session ID/timestamp, lesson/topic/unit performances, recommendation reasons, AI support usage.

## Proposed Improvements

### 1. Enhanced Learning Environment
 1. More realistic learning enviroment, and better update of student performance
 2. More predefined functions to identify learning patterns, or struggle pattern, instead of simply putting minimal processed env info and history into LLM
* **Forgetting curve integration:** Confidence decay, spaced repetition, retention tracking.
* **Dynamic updates:** Adaptive learning rates, context-aware mastery adjustments, scalable difficulty.
* **Advanced pattern detection:** Struggle identification, learning style recognition, progress analysis.


### 2. Advanced Pattern Analysis

* **Specialized analysis functions:** Learning style detection, gap identification, prerequisite mastery tracking, concept mapping.
* **Enhanced LLM inputs:** Structured learning patterns, performance trends, struggle points, and success recognition.

### 3. Enhanced Memory Management
 Enhance memory, more advanced retrieval mechanism instead of simply using the last N session data, 

### 4. Global & Cross-Student Memory

* Maintain a shared memory across multiple students to identify common patterns and best practices.
* Use clustering or embedding-based techniques to match a new student’s profile to similar existing learners, providing tailored recommendations based on earlier successful interventions.

### 5. Prompt Engineering & Model Optimization & System Optimization

* **Individualized testing & A/B experiments:** Systematically test different prompt variants and recommendation strategies on simulated learner profiles to measure which formulations yield the best alignment with ground-truth learning gains.
* **Fine‐tuning and specialized models:** Explore fine‐tuning smaller models on anonymized session data, annotation corpora of learning trajectories, or synthetic learner interactions to reduce reliance on large LLM calls and improve pedagogical accuracy.
* Better visulization of the system decision flow process
  
### 6. Advanced Reasoning Strategies & Learning Theory Integration

* **Multi‐theory decision framework:** Incorporate insights from established learning theories into the LLM’s reasoning chain to explain why a given lesson order supports conceptual mastery.
* **Adaptive reasoning templates:** Build modular reasoning chains—e.g., scaffolding templates that explicitly reference prerequisite mastery, cognitive load constraints, or spaced repetition schedules—so the LLM can select or combine reasoning patterns dynamically.
  
### 7. Knowledge Mapping & Concept Relations

* **Curriculum ontology construction:** Build a structured map of concepts, prerequisites, and learning paths—link each lesson and topic to foundational and advanced concepts.
* **Concept graph integration:** Incorporate a graph-based representation (nodes = concepts, edges = prerequisite relationships) into both the Learning Environment and LLM prompts, allowing more precise gap identification and scaffolding.

### PrismOffline â€” Local Mathematical Problem-Solving Application

## 1. Project Definition

PrismOffline is a local/offline-first mathematical problem-solving application.

The application is intended to provide several ways to work with a mathematical question:

1. **Standard Mode**
2. **Pure Wolfram Mode**
3. **Guided Attempt Mode**

The initial application does **not** require:

* Login
* User accounts
* Cloud accounts
* Workspaces
* Teacher/student management
* Online mode

Those may be considered in a later phase, but they are not part of the current application.

The current priority is to build a strong local mathematical engine and a clean frontend around it.

---

# 2. Core Architecture

```text
                         PrismOffline
                           |
              +------------+------------+
              |                         |
           Frontend                  Backend
              |                         |
              |                    FastAPI
              |                         |
              |                ReasoningService
              |                         |
              |              +----------+----------+
              |              |                     |
              |          Local LLM              Wolfram
              |              |                     |
              |          llama.cpp             Engine 15
              |              |
              |       Current model:
              |       Qwen3-4B-Q4_K_M
              |
              +---- Standard Mode
              +---- Pure Wolfram
              +---- Guided Attempt
```

The frontend should communicate with the backend through HTTP APIs.

The frontend should not directly communicate with Wolfram or llama.cpp.

---

# 3. Backend

## 3.1 Technology

Current backend:

* Python
* FastAPI
* Wolfram Engine
* SymPy provider
* llama.cpp server
* Local GGUF model

Current backend location:

```text
C:\Users\sanhi\Documents\PrismOffline\backend
```

---

# 4. Backend Responsibilities

The backend is responsible for:

* receiving questions
* identifying the requested operation
* normalizing mathematical expressions
* deciding whether Wolfram or the local LLM is appropriate
* executing exact symbolic computation
* generating mathematical explanations
* validating tool calls
* caching successful results
* eventually supporting guided attempts
* providing a stable API to the frontend

The frontend should not contain mathematical execution logic.

---

# 5. Computational Routing

The main routing architecture is:

```text
Question
   |
   v
ReasoningService
   |
   +-- Simple expression
   |       |
   |       +--> Wolfram
   |
   +-- Explicit symbolic operation
   |       |
   |       +--> Wolfram
   |
   +-- Reasoning problem
   |       |
   |       +--> Local LLM
   |
   +-- Complex computational problem
           |
           +--> Local LLM planner
                   |
                   +--> Wolfram
                   |
                   +--> Local LLM explanation
```

The important principle is:

**Use Wolfram for exact symbolic computation and the local LLM for reasoning/explanation.**

The LLM should not unnecessarily perform calculations that Wolfram can perform exactly.

---

# 6. Current Wolfram Architecture

Wolfram Engine 15 is installed locally.

The backend uses a persistent `WolframLanguageSession`.

This is important because starting a new Wolfram kernel for every request creates unnecessary latency.

The persistent session should remain.

Supported operations currently include:

* calculate
* solve
* simplify
* expand
* factor
* differentiate
* integrate
* limit
* plot

---

# 7. Current Local LLM Architecture

The local LLM runs separately through llama.cpp.

Architecture:

```text
FastAPI
   |
   | HTTP
   v
llama-server
   |
   v
GGUF model
```

Current model:

```text
Qwen3-4B-Q4_K_M.gguf
```

Current model location:

```text
models/local/Qwen3-4B-Q4_K_M.gguf
```

Current server:

```text
http://127.0.0.1:8080
```

The current model should remain the default for now.

---

# 8. Local LLM Configuration

The backend should be model-agnostic.

The reasoning service should NOT contain hard-coded logic such as:

```python
if model == "Qwen..."
```

Instead, local LLM configuration should control things such as:

* enabled/disabled
* server URL
* model identifier
* request timeout
* default token budget
* temperature
* thinking mode

The backend should communicate with the local model through the existing `LocalLLMProvider` / `LocalLLMRuntime` abstraction.

The physical GGUF model is loaded by llama.cpp, not by Python.

---

# 9. Model Swapping

Future model testing should be easy.

The desired workflow is:

```text
Change model configuration/startup
            |
            v
Start llama-server with another GGUF
            |
            v
PrismOffline automatically uses it
```

No changes should be required in `ReasoningService`.

The current Qwen model remains the default until another model is deliberately tested.

---

# 10. Thinking Mode

The current Qwen3 setup must use the verified no-thinking behavior for ordinary reasoning requests.

When:

```python
enable_thinking=False
```

the runtime currently uses:

```text
/no_think
```

This substantially reduced unnecessary reasoning latency.

Do not remove this behavior without benchmarking.

---

# 11. LLM Performance

Current measured generation speed is approximately:

```text
9â€“10 tokens/second
```

The model can produce good mathematical explanations, but longer answers naturally take longer.

For example:

```text
128 tokens
â†’ ~12.4 seconds
â†’ answer truncated

160 tokens
â†’ ~15.4 seconds
â†’ answer truncated

199 tokens
â†’ ~22.4 seconds
â†’ complete answer
```

Therefore the project should not impose an unnecessarily low global token limit.

Future improvement:

**Adaptive token budgets depending on the requested task.**

For example:

```text
Short explanation      â†’ smaller budget
Normal proof           â†’ medium budget
Complex derivation     â†’ larger budget
```

This should be implemented only after the core system is stable.

---

# 12. Mathematical Expression Normalization

The backend contains an expression normalizer.

It must correctly preserve mathematical functions such as:

```text
ArcSin
ArcCos
ArcTan
Sin
Cos
Tan
```

while inserting implicit multiplication where appropriate.

For example:

```text
2x       â†’ 2*x
x(x+1)   â†’ x*(x+1)
2Sin[x]  â†’ 2*Sin[x]
ArcSin[x] must remain ArcSin[x]
```

Do not regress this behavior.

---

# 13. Explicit Symbolic Operations

Explicit requests should bypass unnecessary LLM planning.

Examples:

```text
Solve x^2 - 5*x + 6 = 0.

Integrate x^2 with respect to x.

Differentiate x^3 + 2*x.

Factor x^2 - 5*x + 6.

Expand (x+2)(x+3).

Simplify (x^2-1)/(x-1).

Limit (sin(x))/x as x approaches 0.
```

These should directly invoke the corresponding Wolfram operation.

This reduces latency and improves determinism.

---

# 14. Standard Mode

Standard Mode is the main/default user experience.

The user enters:

```text
Question
```

and can choose whether an explanation is wanted.

The frontend should provide something conceptually like:

```text
Question
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                                              â”‚
â”‚ Enter your question...                       â”‚
â”‚                                              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Explanation:
( ) Without explanation
( ) With explanation

                 [ Solve ]
```

The exact UI can be designed later.

---

# 15. Standard Mode â€” Without Explanation

When the user chooses:

```text
Without explanation
```

the backend should return the solution/result with minimal additional text.

For symbolic operations, this should preferably use the deterministic computational path.

Example:

```text
Question:
Integrate x^2 dx

Answer:
xÂ³/3
```

No long LLM explanation should be generated.

---

# 16. Standard Mode â€” With Explanation

When the user chooses:

```text
With explanation
```

the backend should provide:

1. The result
2. A concise mathematical explanation
3. Relevant steps

For example:

```text
Question:
Differentiate xÂ² + 3x

Solution:
2x + 3

Explanation:
Using the power rule...

...
```

The exact presentation should remain clean and mathematical rather than conversationally verbose.

---

# 17. Pure Wolfram Mode

This is a separate mode.

The user explicitly chooses:

```text
Pure Wolfram
```

The purpose is to bypass the LLM completely.

Architecture:

```text
Frontend
   |
   v
/api/solve or dedicated computational endpoint
   |
   v
Wolfram
   |
   v
Solution
```

No LLM planner.

No LLM explanation.

No LLM interpretation unless absolutely necessary.

The goal is:

**maximum deterministic computational behavior.**

The output should contain only the solution/result generated from the computational engine.

Example:

```text
Input:
Solve xÂ² - 5x + 6 = 0

Output:
x = 2, 3
```

The frontend should make it obvious that this mode produces a direct computational solution.

---

# 18. Difference Between Standard and Pure Wolfram

```text
STANDARD
Question
   |
   v
ReasoningService
   |
   +--> computation when appropriate
   |
   +--> reasoning/explanation when requested
   |
   v
Solution

PURE WOLFRAM
Question
   |
   v
Wolfram
   |
   v
Solution
```

Pure Wolfram must not silently invoke the LLM.

---

# 19. Guided Attempt Mode

Guided Attempt Mode should reproduce the useful concept from the earlier Prism AI project, but independently inside PrismOffline.

The purpose is NOT to immediately reveal the complete answer.

Instead:

```text
Question
   |
   v
Guided Attempt
   |
   +--> user attempts
   |
   +--> system evaluates progress
   |
   +--> hint
   |
   +--> another attempt
   |
   +--> more guidance
   |
   v
Final solution
```

---

# 20. Guided Attempt User Experience

Example:

```text
Question

Solve:
xÂ² - 5x + 6 = 0

Your attempt:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                                              â”‚
â”‚                                              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

[ Submit Attempt ]

[ Hint ]
```

After an incorrect or incomplete attempt:

```text
Your attempt:
...

Feedback:
You have identified the correct type of equation.

Hint:
Try finding two numbers whose product is 6
and whose sum is -5.

[ Try Again ]
```

The system should avoid immediately exposing the final answer.

---

# 21. Guided Attempt Progression

Guided attempts should have structured stages.

Possible progression:

```text
Stage 1
Understand the problem

Stage 2
Identify the relevant concept

Stage 3
Choose the method

Stage 4
Perform the next calculation

Stage 5
Complete the solution

Stage 6
Show final solution
```

The actual number of stages should depend on the question.

Simple questions may need only a few stages.

Complex questions may need more.

---

# 22. Guided Attempt Principles

The system should:

* evaluate the student's attempt
* identify what is correct
* identify the next missing step
* provide a useful hint
* avoid unnecessarily revealing later steps
* remember the current attempt state
* allow another attempt
* eventually reveal the complete solution

The final solution should be available when the guided process is completed or the user explicitly requests it.

---

# 23. Guided Attempt Backend Design

Guided Attempt should eventually have a dedicated API rather than abusing `/api/reason`.

Potential structure:

```text
POST /api/guided/start
POST /api/guided/attempt
POST /api/guided/hint
POST /api/guided/solution
```

The exact API can be designed later.

The state should eventually include:

```text
question
current step
attempt history
correct/incorrect status
hints used
completed steps
final solution
```

For the first implementation, this state can remain in memory.

Persistent accounts are not required.

---

# 24. Frontend Modes

Initial frontend navigation should expose three primary modes:

```text
Standard
Pure Wolfram
Guided Attempt
```

The UI should be simple.

A possible structure:

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ PrismOffline                                               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                     â”‚
â”‚  Standard     Pure Wolfram     Guided Attempt       â”‚
â”‚                                                     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                     â”‚
â”‚  Question                                            â”‚
â”‚                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚                                               â”‚  â”‚
â”‚  â”‚                                               â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                     â”‚
â”‚  Options                                            â”‚
â”‚                                                     â”‚
â”‚  â—‹ Without explanation                              â”‚
â”‚  â—‹ With explanation                                 â”‚
â”‚                                                     â”‚
â”‚                         [ Solve ]                    â”‚
â”‚                                                     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Solution                                            â”‚
â”‚                                                     â”‚
â”‚  ...                                                 â”‚
â”‚                                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

# 25. Frontend Design Principles

The frontend should be:

* clean
* desktop-friendly
* responsive
* mathematical
* minimal
* easy to understand
* focused on the question and solution

Avoid unnecessary UI complexity.

No login page initially.

No account management initially.

No workspace UI.

No teacher/student UI.

No cloud settings.

No provider/model names in the user interface.

---

# 26. Frontend API Layer

The frontend should communicate with the backend through a dedicated API service layer.

Do not scatter raw `fetch()` calls throughout React components.

Conceptually:

```text
src/
    services/
        api.ts

    components/
        ...

    modes/
        StandardMode
        WolframMode
        GuidedAttemptMode
```

The exact structure can be changed after inspecting the actual frontend project.

---

# 27. Standard Mode API

Possible initial endpoint:

```text
POST /api/reason
```

Request:

```json
{
  "query": "Differentiate x^2 + 3x",
  "explanation": true
}
```

The backend should eventually support an explicit explanation preference rather than requiring the frontend to infer it.

Possible values:

```text
explanation: true
explanation: false
```

If the existing request schema does not support this yet, add it cleanly.

---

# 28. Pure Wolfram API

Pure Wolfram should preferably use the existing computational API rather than duplicating execution logic.

Possible request:

```text
POST /api/solve
```

Example:

```json
{
  "query": "Integrate[x^2, x]"
}
```

The frontend should present only the final computational result.

---

# 29. Explanation Generation

When explanation is requested, the backend should distinguish between:

### Exact computation

Use Wolfram first.

### Explanation

Use the local LLM only when necessary.

For example:

```text
Differentiate xÂ² + 3x

Wolfram:
2x + 3

LLM:
Explain the mathematical reasoning concisely.
```

This avoids asking the LLM to calculate the answer independently when an exact engine already produced it.

---

# 30. Error Handling

The frontend should display useful errors.

Examples:

```text
Unable to solve this question.

The computational engine is currently unavailable.

The local reasoning engine is currently unavailable.

The question could not be interpreted.
```

Do not expose:

* stack traces
* Python exceptions
* internal provider details
* model paths
* llama.cpp errors

to ordinary users.

Developer logs can contain detailed diagnostics.

---

# 31. No Authentication Initially

PrismOffline initially has:

```text
No login
No registration
No accounts
No passwords
No sessions tied to users
```

The application should open directly into the solver.

Future authentication can be added if the application eventually needs:

* saved history
* personal settings
* synchronized data
* accounts
* cloud storage

But none of this should block the current application.

---

# 32. Future Account Support

Accounts are a future possibility only.

Potential future features:

```text
Login
Registration
User settings
Saved solutions
History
Bookmarks
Personal model settings
```

Do not build this now.

The current architecture should simply avoid making future accounts impossible.

---

# 33. Future Conversation History

Later, PrismOffline may support:

```text
Question
   â†“
Solution
   â†“
Follow-up
   â†“
Additional explanation
   â†“
New computation
```

This requires conversation context.

Do not implement persistent conversation history until the basic modes are stable.

---

# 34. Future Input Types

Later, the application may accept:

* typed mathematics
* natural-language questions
* images
* scanned questions
* PDFs
* diagrams

The initial version can focus on typed text.

---

# 35. Future Graphing

Wolfram already supports plotting.

Eventually Standard Mode may display:

```text
Solution
+
Graph
```

when the question requires one.

Pure Wolfram Mode may return graph data directly.

The frontend should eventually have a reusable graph component.

---

# 36. Testing Strategy

Every backend change should be tested at three levels.

## Unit tests

Test:

* normalization
* operation detection
* validation
* routing
* configuration

## Provider tests

Test:

* Wolfram
* local LLM
* SymPy

## API tests

Test:

```text
/api/reason
/api/solve
```

and later:

```text
/api/guided/*
```

---

# 37. Benchmarking

When testing local models, use a fixed benchmark suite.

Include:

```text
Simple proof
Algebra
Derivative
Integral
Inequality
Word problem
Multi-step reasoning
Ambiguous problem
```

Record:

```text
Model
Quantization
Prompt tokens
Completion tokens
Latency
Tokens/sec
Correctness
Completeness
```

Do not choose a model based purely on tokens/sec.

---

# 38. Current Development Priority

The development order should be:

## Phase 1 â€” Backend foundation

Complete and stabilize:

* FastAPI
* Wolfram
* SymPy
* local LLM
* provider abstraction
* expression normalization
* deterministic symbolic routing
* reasoning routing
* caching
* error handling

## Phase 2 â€” Local LLM configuration

Make:

* endpoint
* model identifier
* generation settings
* thinking setting
* timeout

configurable.

Keep Qwen3-4B as the current default.

## Phase 3 â€” Basic frontend

Build:

* Standard Mode
* question input
* explanation toggle
* solution display
* loading state
* error state

## Phase 4 â€” Pure Wolfram Mode

Build:

* direct computational input
* direct result
* no LLM

## Phase 5 â€” Guided Attempt

Build:

* question
* attempt
* feedback
* hints
* progression
* final solution

## Phase 6 â€” Mathematical UX

Add:

* proper math rendering
* equations
* graphs
* structured steps
* copy solution
* clear/reset controls

## Phase 7 â€” Optimization

Only after functionality is stable:

* adaptive token budgets
* model benchmarking
* response caching improvements
* inference optimization
* hardware optimization if necessary

## Phase 8 â€” Optional future account system

Only if eventually needed:

* accounts
* login
* saved history
* personalization

---

# 39. Current Project Boundary

PrismOffline must remain independent from the older Prism AI application.

Do NOT import Prism AI's:

* authentication system
* workspace system
* teacher/student architecture
* online mode
* cloud architecture
* existing frontend
* account system

Guided Attempt is the one major concept being reused conceptually from that project.

It should be implemented natively for PrismOffline rather than copying unrelated Prism AI architecture.

---

# 40. Current State â€” Known Working

The following are currently working:

* persistent Wolfram session
* symbolic solve
* integrate
* differentiate
* factor
* expand
* simplify
* limit
* expression normalization
* ArcSin/ArcCos/ArcTan handling
* deterministic explicit operation routing
* local LLM reasoning
* `/no_think`
* local LLM HTTP runtime
* planner path
* undefined-function protection
* result caching

Current local model:

```text
Qwen3-4B-Q4_K_M.gguf
```

Keep it.

---

# 41. Immediate Next Development Task

Before building the frontend, finish the local LLM configuration cleanup.

Inspect:

```text
backend/providers/local_llm/config.py
backend/providers/local_llm/provider.py
backend/providers/local_llm/runtime.py
backend/services/provider_router.py
backend/services/reasoning_service.py
run_nova.ps1
```

Make the configuration clean enough that another local model can be tested later without rewriting the reasoning architecture.

Do not change:

* current Qwen model
* `/no_think`
* Wolfram architecture
* explicit operation routing
* reasoning behavior
* frontend yet

After that, begin the frontend.

---

# 42. Frontend Immediate Task

Build the first frontend around the three modes:

```text
STANDARD
PURE WOLFRAM
GUIDED ATTEMPT
```

Standard Mode initially contains:

```text
Question input

Explanation:
[ With explanation ]
[ Without explanation ]

[ Solve ]
```

The result area should support:

```text
Solution
Explanation
Steps
```

depending on the selected option.

Pure Wolfram should provide:

```text
Question
[ Solve ]
Solution
```

with no LLM explanation.

Guided Attempt should provide:

```text
Question
Attempt input
[ Submit Attempt ]
[ Hint ]
Feedback
Progress
Final Solution
```

---

# 43. Guiding Principle

PrismOffline should remain simple at the surface while being sophisticated internally.

The user should experience:

```text
Ask a question
    â†“
Choose how to solve it
    â†“
Get the result
```

The internal system can decide whether that means:

```text
Wolfram
Local LLM
LLM + Wolfram
Guided reasoning
```

The implementation should preserve that separation.

---

# 44. Non-Goals for the Current Version

Do NOT build:

* authentication
* accounts
* cloud synchronization
* workspaces
* teachers
* students
* online mode
* external AI APIs
* provider selection exposed to users
* model selection exposed to ordinary users

These are outside the current PrismOffline scope.


# PrismOffline â€” Local Mathematical Problem-Solving Application

## 1. Project Definition

PrismOffline is a local/offline-first mathematical problem-solving application.

The application is intended to provide several ways to work with a mathematical question:

1. **Standard Mode**
2. **Pure Wolfram Mode**
3. **Guided Attempt Mode**

The initial application does **not** require:

* Login
* User accounts
* Cloud accounts
* Workspaces
* Teacher/student management
* Online mode

Those may be considered in a later phase, but they are not part of the current application.

The current priority is to build a strong local mathematical engine and a clean frontend around it.

---

# 2. Core Architecture

```text
                         PrismOffline
                           |
              +------------+------------+
              |                         |
           Frontend                  Backend
              |                         |
              |                    FastAPI
              |                         |
              |                ReasoningService
              |                         |
              |              +----------+----------+
              |              |                     |
              |          Local LLM              Wolfram
              |              |                     |
              |          llama.cpp             Engine 15
              |              |
              |       Current model:
              |       Qwen3-4B-Q4_K_M
              |
              +---- Standard Mode
              +---- Pure Wolfram
              +---- Guided Attempt
```

The frontend should communicate with the backend through HTTP APIs.

The frontend should not directly communicate with Wolfram or llama.cpp.

---

# 3. Backend

## 3.1 Technology

Current backend:

* Python
* FastAPI
* Wolfram Engine
* SymPy provider
* llama.cpp server
* Local GGUF model

Current backend location:

```text
C:\Users\sanhi\Documents\PrismOffline\backend
```

---

# 4. Backend Responsibilities

The backend is responsible for:

* receiving questions
* identifying the requested operation
* normalizing mathematical expressions
* deciding whether Wolfram or the local LLM is appropriate
* executing exact symbolic computation
* generating mathematical explanations
* validating tool calls
* caching successful results
* eventually supporting guided attempts
* providing a stable API to the frontend

The frontend should not contain mathematical execution logic.

---

# 5. Computational Routing

The main routing architecture is:

```text
Question
   |
   v
ReasoningService
   |
   +-- Simple expression
   |       |
   |       +--> Wolfram
   |
   +-- Explicit symbolic operation
   |       |
   |       +--> Wolfram
   |
   +-- Reasoning problem
   |       |
   |       +--> Local LLM
   |
   +-- Complex computational problem
           |
           +--> Local LLM planner
                   |
                   +--> Wolfram
                   |
                   +--> Local LLM explanation
```

The important principle is:

**Use Wolfram for exact symbolic computation and the local LLM for reasoning/explanation.**

The LLM should not unnecessarily perform calculations that Wolfram can perform exactly.

---

# 6. Current Wolfram Architecture

Wolfram Engine 15 is installed locally.

The backend uses a persistent `WolframLanguageSession`.

This is important because starting a new Wolfram kernel for every request creates unnecessary latency.

The persistent session should remain.

Supported operations currently include:

* calculate
* solve
* simplify
* expand
* factor
* differentiate
* integrate
* limit
* plot

---

# 7. Current Local LLM Architecture

The local LLM runs separately through llama.cpp.

Architecture:

```text
FastAPI
   |
   | HTTP
   v
llama-server
   |
   v
GGUF model
```

Current model:

```text
Qwen3-4B-Q4_K_M.gguf
```

Current model location:

```text
models/local/Qwen3-4B-Q4_K_M.gguf
```

Current server:

```text
http://127.0.0.1:8080
```

The current model should remain the default for now.

---

# 8. Local LLM Configuration

The backend should be model-agnostic.

The reasoning service should NOT contain hard-coded logic such as:

```python
if model == "Qwen..."
```

Instead, local LLM configuration should control things such as:

* enabled/disabled
* server URL
* model identifier
* request timeout
* default token budget
* temperature
* thinking mode

The backend should communicate with the local model through the existing `LocalLLMProvider` / `LocalLLMRuntime` abstraction.

The physical GGUF model is loaded by llama.cpp, not by Python.

---

# 9. Model Swapping

Future model testing should be easy.

The desired workflow is:

```text
Change model configuration/startup
            |
            v
Start llama-server with another GGUF
            |
            v
PrismOffline automatically uses it
```

No changes should be required in `ReasoningService`.

The current Qwen model remains the default until another model is deliberately tested.

---

# 10. Thinking Mode

The current Qwen3 setup must use the verified no-thinking behavior for ordinary reasoning requests.

When:

```python
enable_thinking=False
```

the runtime currently uses:

```text
/no_think
```

This substantially reduced unnecessary reasoning latency.

Do not remove this behavior without benchmarking.

---

# 11. LLM Performance

Current measured generation speed is approximately:

```text
9â€“10 tokens/second
```

The model can produce good mathematical explanations, but longer answers naturally take longer.

For example:

```text
128 tokens
â†’ ~12.4 seconds
â†’ answer truncated

160 tokens
â†’ ~15.4 seconds
â†’ answer truncated

199 tokens
â†’ ~22.4 seconds
â†’ complete answer
```

Therefore the project should not impose an unnecessarily low global token limit.

Future improvement:

**Adaptive token budgets depending on the requested task.**

For example:

```text
Short explanation      â†’ smaller budget
Normal proof           â†’ medium budget
Complex derivation     â†’ larger budget
```

This should be implemented only after the core system is stable.

---

# 12. Mathematical Expression Normalization

The backend contains an expression normalizer.

It must correctly preserve mathematical functions such as:

```text
ArcSin
ArcCos
ArcTan
Sin
Cos
Tan
```

while inserting implicit multiplication where appropriate.

For example:

```text
2x       â†’ 2*x
x(x+1)   â†’ x*(x+1)
2Sin[x]  â†’ 2*Sin[x]
ArcSin[x] must remain ArcSin[x]
```

Do not regress this behavior.

---

# 13. Explicit Symbolic Operations

Explicit requests should bypass unnecessary LLM planning.

Examples:

```text
Solve x^2 - 5*x + 6 = 0.

Integrate x^2 with respect to x.

Differentiate x^3 + 2*x.

Factor x^2 - 5*x + 6.

Expand (x+2)(x+3).

Simplify (x^2-1)/(x-1).

Limit (sin(x))/x as x approaches 0.
```

These should directly invoke the corresponding Wolfram operation.

This reduces latency and improves determinism.

---

# 14. Standard Mode

Standard Mode is the main/default user experience.

The user enters:

```text
Question
```

and can choose whether an explanation is wanted.

The frontend should provide something conceptually like:

```text
Question
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                                              â”‚
â”‚ Enter your question...                       â”‚
â”‚                                              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

Explanation:
( ) Without explanation
( ) With explanation

                 [ Solve ]
```

The exact UI can be designed later.

---

# 15. Standard Mode â€” Without Explanation

When the user chooses:

```text
Without explanation
```

the backend should return the solution/result with minimal additional text.

For symbolic operations, this should preferably use the deterministic computational path.

Example:

```text
Question:
Integrate x^2 dx

Answer:
xÂ³/3
```

No long LLM explanation should be generated.

---

# 16. Standard Mode â€” With Explanation

When the user chooses:

```text
With explanation
```

the backend should provide:

1. The result
2. A concise mathematical explanation
3. Relevant steps

For example:

```text
Question:
Differentiate xÂ² + 3x

Solution:
2x + 3

Explanation:
Using the power rule...

...
```

The exact presentation should remain clean and mathematical rather than conversationally verbose.

---

# 17. Pure Wolfram Mode

This is a separate mode.

The user explicitly chooses:

```text
Pure Wolfram
```

The purpose is to bypass the LLM completely.

Architecture:

```text
Frontend
   |
   v
/api/solve or dedicated computational endpoint
   |
   v
Wolfram
   |
   v
Solution
```

No LLM planner.

No LLM explanation.

No LLM interpretation unless absolutely necessary.

The goal is:

**maximum deterministic computational behavior.**

The output should contain only the solution/result generated from the computational engine.

Example:

```text
Input:
Solve xÂ² - 5x + 6 = 0

Output:
x = 2, 3
```

The frontend should make it obvious that this mode produces a direct computational solution.

---

# 18. Difference Between Standard and Pure Wolfram

```text
STANDARD
Question
   |
   v
ReasoningService
   |
   +--> computation when appropriate
   |
   +--> reasoning/explanation when requested
   |
   v
Solution

PURE WOLFRAM
Question
   |
   v
Wolfram
   |
   v
Solution
```

Pure Wolfram must not silently invoke the LLM.

---

# 19. Guided Attempt Mode

Guided Attempt Mode should reproduce the useful concept from the earlier Prism AI project, but independently inside PrismOffline.

The purpose is NOT to immediately reveal the complete answer.

Instead:

```text
Question
   |
   v
Guided Attempt
   |
   +--> user attempts
   |
   +--> system evaluates progress
   |
   +--> hint
   |
   +--> another attempt
   |
   +--> more guidance
   |
   v
Final solution
```

---

# 20. Guided Attempt User Experience

Example:

```text
Question

Solve:
xÂ² - 5x + 6 = 0

Your attempt:
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                                              â”‚
â”‚                                              â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

[ Submit Attempt ]

[ Hint ]
```

After an incorrect or incomplete attempt:

```text
Your attempt:
...

Feedback:
You have identified the correct type of equation.

Hint:
Try finding two numbers whose product is 6
and whose sum is -5.

[ Try Again ]
```

The system should avoid immediately exposing the final answer.

---

# 21. Guided Attempt Progression

Guided attempts should have structured stages.

Possible progression:

```text
Stage 1
Understand the problem

Stage 2
Identify the relevant concept

Stage 3
Choose the method

Stage 4
Perform the next calculation

Stage 5
Complete the solution

Stage 6
Show final solution
```

The actual number of stages should depend on the question.

Simple questions may need only a few stages.

Complex questions may need more.

---

# 22. Guided Attempt Principles

The system should:

* evaluate the student's attempt
* identify what is correct
* identify the next missing step
* provide a useful hint
* avoid unnecessarily revealing later steps
* remember the current attempt state
* allow another attempt
* eventually reveal the complete solution

The final solution should be available when the guided process is completed or the user explicitly requests it.

---

# 23. Guided Attempt Backend Design

Guided Attempt should eventually have a dedicated API rather than abusing `/api/reason`.

Potential structure:

```text
POST /api/guided/start
POST /api/guided/attempt
POST /api/guided/hint
POST /api/guided/solution
```

The exact API can be designed later.

The state should eventually include:

```text
question
current step
attempt history
correct/incorrect status
hints used
completed steps
final solution
```

For the first implementation, this state can remain in memory.

Persistent accounts are not required.

---

# 24. Frontend Modes

Initial frontend navigation should expose three primary modes:

```text
Standard
Pure Wolfram
Guided Attempt
```

The UI should be simple.

A possible structure:

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ PrismOffline                                               â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                     â”‚
â”‚  Standard     Pure Wolfram     Guided Attempt       â”‚
â”‚                                                     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                     â”‚
â”‚  Question                                            â”‚
â”‚                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚                                               â”‚  â”‚
â”‚  â”‚                                               â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                     â”‚
â”‚  Options                                            â”‚
â”‚                                                     â”‚
â”‚  â—‹ Without explanation                              â”‚
â”‚  â—‹ With explanation                                 â”‚
â”‚                                                     â”‚
â”‚                         [ Solve ]                    â”‚
â”‚                                                     â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  Solution                                            â”‚
â”‚                                                     â”‚
â”‚  ...                                                 â”‚
â”‚                                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

# 25. Frontend Design Principles

The frontend should be:

* clean
* desktop-friendly
* responsive
* mathematical
* minimal
* easy to understand
* focused on the question and solution

Avoid unnecessary UI complexity.

No login page initially.

No account management initially.

No workspace UI.

No teacher/student UI.

No cloud settings.

No provider/model names in the user interface.

---

# 26. Frontend API Layer

The frontend should communicate with the backend through a dedicated API service layer.

Do not scatter raw `fetch()` calls throughout React components.

Conceptually:

```text
src/
    services/
        api.ts

    components/
        ...

    modes/
        StandardMode
        WolframMode
        GuidedAttemptMode
```

The exact structure can be changed after inspecting the actual frontend project.

---

# 27. Standard Mode API

Possible initial endpoint:

```text
POST /api/reason
```

Request:

```json
{
  "query": "Differentiate x^2 + 3x",
  "explanation": true
}
```

The backend should eventually support an explicit explanation preference rather than requiring the frontend to infer it.

Possible values:

```text
explanation: true
explanation: false
```

If the existing request schema does not support this yet, add it cleanly.

---

# 28. Pure Wolfram API

Pure Wolfram should preferably use the existing computational API rather than duplicating execution logic.

Possible request:

```text
POST /api/solve
```

Example:

```json
{
  "query": "Integrate[x^2, x]"
}
```

The frontend should present only the final computational result.

---

# 29. Explanation Generation

When explanation is requested, the backend should distinguish between:

### Exact computation

Use Wolfram first.

### Explanation

Use the local LLM only when necessary.

For example:

```text
Differentiate xÂ² + 3x

Wolfram:
2x + 3

LLM:
Explain the mathematical reasoning concisely.
```

This avoids asking the LLM to calculate the answer independently when an exact engine already produced it.

---

# 30. Error Handling

The frontend should display useful errors.

Examples:

```text
Unable to solve this question.

The computational engine is currently unavailable.

The local reasoning engine is currently unavailable.

The question could not be interpreted.
```

Do not expose:

* stack traces
* Python exceptions
* internal provider details
* model paths
* llama.cpp errors

to ordinary users.

Developer logs can contain detailed diagnostics.

---

# 31. No Authentication Initially

PrismOffline initially has:

```text
No login
No registration
No accounts
No passwords
No sessions tied to users
```

The application should open directly into the solver.

Future authentication can be added if the application eventually needs:

* saved history
* personal settings
* synchronized data
* accounts
* cloud storage

But none of this should block the current application.

---

# 32. Future Account Support

Accounts are a future possibility only.

Potential future features:

```text
Login
Registration
User settings
Saved solutions
History
Bookmarks
Personal model settings
```

Do not build this now.

The current architecture should simply avoid making future accounts impossible.

---

# 33. Future Conversation History

Later, PrismOffline may support:

```text
Question
   â†“
Solution
   â†“
Follow-up
   â†“
Additional explanation
   â†“
New computation
```

This requires conversation context.

Do not implement persistent conversation history until the basic modes are stable.

---

# 34. Future Input Types

Later, the application may accept:

* typed mathematics
* natural-language questions
* images
* scanned questions
* PDFs
* diagrams

The initial version can focus on typed text.

---

# 35. Future Graphing

Wolfram already supports plotting.

Eventually Standard Mode may display:

```text
Solution
+
Graph
```

when the question requires one.

Pure Wolfram Mode may return graph data directly.

The frontend should eventually have a reusable graph component.

---

# 36. Testing Strategy

Every backend change should be tested at three levels.

## Unit tests

Test:

* normalization
* operation detection
* validation
* routing
* configuration

## Provider tests

Test:

* Wolfram
* local LLM
* SymPy

## API tests

Test:

```text
/api/reason
/api/solve
```

and later:

```text
/api/guided/*
```

---

# 37. Benchmarking

When testing local models, use a fixed benchmark suite.

Include:

```text
Simple proof
Algebra
Derivative
Integral
Inequality
Word problem
Multi-step reasoning
Ambiguous problem
```

Record:

```text
Model
Quantization
Prompt tokens
Completion tokens
Latency
Tokens/sec
Correctness
Completeness
```

Do not choose a model based purely on tokens/sec.

---

# 38. Current Development Priority

The development order should be:

## Phase 1 â€” Backend foundation

Complete and stabilize:

* FastAPI
* Wolfram
* SymPy
* local LLM
* provider abstraction
* expression normalization
* deterministic symbolic routing
* reasoning routing
* caching
* error handling

## Phase 2 â€” Local LLM configuration

Make:

* endpoint
* model identifier
* generation settings
* thinking setting
* timeout

configurable.

Keep Qwen3-4B as the current default.

## Phase 3 â€” Basic frontend

Build:

* Standard Mode
* question input
* explanation toggle
* solution display
* loading state
* error state

## Phase 4 â€” Pure Wolfram Mode

Build:

* direct computational input
* direct result
* no LLM

## Phase 5 â€” Guided Attempt

Build:

* question
* attempt
* feedback
* hints
* progression
* final solution

## Phase 6 â€” Mathematical UX

Add:

* proper math rendering
* equations
* graphs
* structured steps
* copy solution
* clear/reset controls

## Phase 7 â€” Optimization

Only after functionality is stable:

* adaptive token budgets
* model benchmarking
* response caching improvements
* inference optimization
* hardware optimization if necessary

## Phase 8 â€” Optional future account system

Only if eventually needed:

* accounts
* login
* saved history
* personalization

---

# 39. Current Project Boundary

PrismOffline must remain independent from the older Prism AI application.

Do NOT import Prism AI's:

* authentication system
* workspace system
* teacher/student architecture
* online mode
* cloud architecture
* existing frontend
* account system

Guided Attempt is the one major concept being reused conceptually from that project.

It should be implemented natively for PrismOffline rather than copying unrelated Prism AI architecture.

---

# 40. Current State â€” Known Working

The following are currently working:

* persistent Wolfram session
* symbolic solve
* integrate
* differentiate
* factor
* expand
* simplify
* limit
* expression normalization
* ArcSin/ArcCos/ArcTan handling
* deterministic explicit operation routing
* local LLM reasoning
* `/no_think`
* local LLM HTTP runtime
* planner path
* undefined-function protection
* result caching

Current local model:

```text
Qwen3-4B-Q4_K_M.gguf
```

Keep it.

---

# 41. Immediate Next Development Task

Before building the frontend, finish the local LLM configuration cleanup.

Inspect:

```text
backend/providers/local_llm/config.py
backend/providers/local_llm/provider.py
backend/providers/local_llm/runtime.py
backend/services/provider_router.py
backend/services/reasoning_service.py
run_nova.ps1
```

Make the configuration clean enough that another local model can be tested later without rewriting the reasoning architecture.

Do not change:

* current Qwen model
* `/no_think`
* Wolfram architecture
* explicit operation routing
* reasoning behavior
* frontend yet

After that, begin the frontend.

---

# 42. Frontend Immediate Task

Build the first frontend around the three modes:

```text
STANDARD
PURE WOLFRAM
GUIDED ATTEMPT
```

Standard Mode initially contains:

```text
Question input

Explanation:
[ With explanation ]
[ Without explanation ]

[ Solve ]
```

The result area should support:

```text
Solution
Explanation
Steps
```

depending on the selected option.

Pure Wolfram should provide:

```text
Question
[ Solve ]
Solution
```

with no LLM explanation.

Guided Attempt should provide:

```text
Question
Attempt input
[ Submit Attempt ]
[ Hint ]
Feedback
Progress
Final Solution
```

---

# 43. Guiding Principle

PrismOffline should remain simple at the surface while being sophisticated internally.

The user should experience:

```text
Ask a question
    â†“
Choose how to solve it
    â†“
Get the result
```

The internal system can decide whether that means:

```text
Wolfram
Local LLM
LLM + Wolfram
Guided reasoning
```

The implementation should preserve that separation.

---

# 44. Non-Goals for the Current Version

Do NOT build:

* authentication
* accounts
* cloud synchronization
* workspaces
* teachers
* students
* online mode
* external AI APIs
* provider selection exposed to users
* model selection exposed to ordinary users

These are outside the current PrismOffline scope.


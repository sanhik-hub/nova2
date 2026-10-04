# PrismOffline

Local/offline-first STEM problem solving with a FastAPI backend, React/Vite frontend, local Qwen3 GGUF inference through llama.cpp, and a persistent Wolfram Engine session.

> Windows is the current supported development environment.

## 1. Clone and install

~~~powershell
git clone https://github.com/sanhik-hub/nova2.git
Set-Location .\nova2
~~~

Install:

- Git: https://git-scm.com/download/win
- Python 3.13: https://www.python.org/downloads/windows/
- Node.js: https://nodejs.org/en/download
- llama.cpp: https://github.com/ggml-org/llama.cpp/releases
- Qwen3 GGUF: https://huggingface.co/Qwen/Qwen3-4B-GGUF
- Wolfram Engine: https://www.wolfram.com/engine/

Create the Python environment:

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn sympy wolframclient
~~~

The repository currently does not contain a checked-in requirements.txt.

## 2. Install llama.cpp

Download a Windows build from the official releases page:

https://github.com/ggml-org/llama.cpp/releases

For a CPU-only machine use the Windows x64 CPU build. GPU-specific builds are available for CUDA, Vulkan, OpenVINO and other backends.

Place the server executable here:

~~~text
nova2/
└── engines/
    └── llama.cpp/
        └── llama-server.exe
~~~

The launcher expects that exact path.

Official installation documentation:

https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md

## 3. Install the Qwen3 model

The current default model is:

~~~text
Qwen3-4B-Q4_K_M.gguf
~~~

Download it from the official Qwen repository:

https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf

Place it here:

~~~text
nova2/
└── models/
    └── local/
        └── Qwen3-4B-Q4_K_M.gguf
~~~

The default configuration is in:

~~~text
backend/providers/local_llm/config.py
~~~

Default model path:

~~~text
models/local/Qwen3-4B-Q4_K_M.gguf
~~~

You can override it with LOCAL_LLM_MODEL.

~~~powershell
$env:LOCAL_LLM_MODEL = "D:\Models\Qwen3-4B-Q4_K_M.gguf"
~~~

The backend talks to llama.cpp over HTTP. Python does not load the GGUF directly.

Default llama.cpp server:

~~~text
http://127.0.0.1:8080
~~~

## 4. Install and configure Wolfram Engine

Open Wolfram's official Windows setup guide:

https://support.wolfram.com/46069

Also see:

- Wolfram Engine: https://www.wolfram.com/engine/
- WolframScript installation: https://reference.wolfram.com/language/workflow/InstallWolframScript.html.en

Install Wolfram Engine 15, activate it with a Wolfram ID, and verify WolframScript.

~~~powershell
wolframscript --version
~~~

The current backend expects the normal Windows installation:

~~~text
C:\Program Files\Wolfram Research\
├── WolframScript\
│   └── wolframscript.exe
└── Wolfram Engine\
    └── 15.0\
        └── WolframKernel.exe
~~~

These defaults are defined in:

~~~text
backend/services/wolfram_engine.py
~~~

If Wolfram Engine is installed somewhere else or a different major version is used, update the configured kernel path in that file.

The backend creates one persistent WolframLanguageSession during FastAPI startup and closes it during shutdown. A small 2 + 2 evaluation is used to verify startup.

## 5. Start the application

The launcher is:

~~~text
run_nova.ps1
~~~

It currently contains a machine-specific Python path. For a fresh clone, change the Python line to this clone's environment:

~~~powershell
$Python = (Resolve-Path ".\.venv\Scripts\python.exe").Path
~~~

Then run:

~~~powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_nova.ps1
~~~

The launcher starts:

1. llama.cpp
2. FastAPI
3. Vite
4. the persistent Wolfram session through the backend

Default addresses:

~~~text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:8000
LLM:      http://127.0.0.1:8080
Docs:     http://127.0.0.1:8000/docs
~~~

### Manual startup

llama.cpp:

~~~powershell
.\engines\llama.cpp\llama-server.exe -m ".\models\local\Qwen3-4B-Q4_K_M.gguf" --host 127.0.0.1 --port 8080 --ctx-size 32768
~~~

Backend:

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
~~~

Frontend:

~~~powershell
Set-Location .\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
~~~

## 6. Current application

### Standard Mode

The main problem-solving interface for:

- Mathematics
- Physics
- Chemistry
- General science

It supports result-only answers, worked explanations, and unlimited consecutive follow-ups. Each follow-up receives the original question, original answer, and previous follow-up history as context.

Main endpoints:

~~~text
POST /api/reason
POST /api/follow-up
~~~

### Guided Attempt

Interactive step-by-step solving with:

- generated steps and MCQs
- hints
- answer tracking
- scoring
- reattempts
- optional complete solution
- saved attempts

Endpoints:

~~~text
POST /api/guided/start
POST /api/guided/solution
~~~

### Exact computation

Wolfram is the authoritative computational engine for supported operations such as:

- calculate/evaluate
- solve
- simplify
- expand
- factor
- differentiate
- integrate
- limits
- plotting

Relevant code:

~~~text
backend/services/computation_service.py
backend/services/wolfram_engine.py
backend/providers/wolfram/
~~~

### Local reasoning

The local model handles natural-language STEM interpretation, reasoning, explanations, Guided Attempt generation, and follow-ups.

Relevant code:

~~~text
backend/providers/local_llm/
backend/services/reasoning_service.py
~~~

### Scientific rendering

The frontend renders mathematical and chemical notation, including subscripts, explicit bonds, formal charges, reaction arrows, and lone-pair notation.

Relevant components:

~~~text
frontend/src/components/MathText.tsx
frontend/src/components/WorkspaceSolutionText.tsx
~~~

### Local accounts and saved work

Accounts and saved work use a local SQLite database:

~~~text
%LOCALAPPDATA%\PrismOffline\prismoffline.db
~~~

The system supports saved solutions, Guided Attempts, save-by-default, and saving follow-up questions/answers with the original question.

Relevant code:

~~~text
backend/services/auth_storage.py
frontend/src/components/AuthView.tsx
frontend/src/components/AccountView.tsx
~~~

## 7. Current project structure

~~~text
nova2/
├── backend/
│   ├── main.py
│   ├── providers/
│   │   ├── local_llm/
│   │   ├── planner_llm/
│   │   ├── sympy/
│   │   └── wolfram/
│   ├── schemas/
│   └── services/
│       ├── auth_storage.py
│       ├── computation_service.py
│       ├── execution_service.py
│       ├── expression_normalizer.py
│       ├── guided_attempt_service.py
│       ├── provider_router.py
│       ├── reasoning_service.py
│       ├── wolfram_engine.py
│       ├── language/
│       └── learning/
│
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.tsx
│       ├── App.css
│       ├── components/
│       ├── modes/
│       └── services/
│
├── control/
│   ├── server.py
│   └── start_control.ps1
│
├── tests/
│   ├── test_api.py
│   └── test_wolfram.py
│
├── tools/
│   ├── benchmark_llama.py
│   └── test_planner.py
│
├── run_nova.ps1
├── readme.md
├── engines/
│   └── llama.cpp/
│       └── llama-server.exe
└── models/
    └── local/
        └── Qwen3-4B-Q4_K_M.gguf
~~~

The engines/ and models/ directories are local runtime dependencies and may need to be created after cloning.

## 8. Important files

| File | Purpose |
|---|---|
| backend/main.py | FastAPI application and API routes |
| backend/services/reasoning_service.py | STEM reasoning and explanation orchestration |
| backend/services/computation_service.py | Computational operations |
| backend/services/execution_service.py | Parsing, routing and execution |
| backend/services/expression_normalizer.py | Mathematical expression normalization |
| backend/services/guided_attempt_service.py | Guided Attempt generation and validation |
| backend/services/wolfram_engine.py | Persistent Wolfram Engine integration |
| backend/providers/local_llm/config.py | Local model/server configuration |
| backend/providers/local_llm/runtime.py | HTTP client for llama.cpp |
| backend/services/auth_storage.py | Local accounts and saved work |
| frontend/src/modes/StandardMode.tsx | Main question-solving UI |
| frontend/src/services/api.ts | Frontend/backend API layer |
| frontend/src/components/MathText.tsx | Math/chemistry rendering |
| run_nova.ps1 | One-command local startup |

## 9. Backend API overview

| Endpoint | Purpose |
|---|---|
| POST /api/auth/register | Create local account |
| POST /api/auth/login | Login |
| POST /api/auth/logout | Logout |
| GET /api/auth/me | Current account |
| GET /api/settings | Read settings |
| PUT /api/settings | Update settings |
| GET /api/storage | List saved work |
| POST /api/storage | Save work |
| PUT /api/storage/{id} | Update saved work |
| DELETE /api/storage/{id} | Delete saved work |
| POST /api/reason | Solve a STEM question |
| POST /api/follow-up | Answer a question-session follow-up |
| POST /api/guided/start | Start Guided Attempt |
| POST /api/guided/solution | Generate final Guided solution |
| GET /api/health | Backend/Wolfram health |
| POST /api/wolfram/start | Start and verify Wolfram |
| POST /api/wolfram/restart | Restart Wolfram session |
| POST /api/plot | Generate plot data |

## 10. Development principles

- Frontend communicates with the backend; it does not directly call Wolfram or llama.cpp.
- Wolfram is used for authoritative exact computation where applicable.
- llama.cpp owns the local GGUF model process.
- The project is offline-first after the required software, model and runtime components are installed.
- Token estimates are advisory only; there is no hard global answer-token ceiling.
- Provider/model implementation details are kept out of the user-facing application UI.
- Chemical and scientific formatting should remain unambiguous.
- The persistent Wolfram session should be preserved rather than launching a new kernel for every request.

## 11. Verification

After startup:

~~~text
http://127.0.0.1:8080/health
http://127.0.0.1:8000/api/health
http://127.0.0.1:5173
~~~

Try a basic calculation:

~~~text
Differentiate x^2 + 3x
~~~

Expected result:

~~~text
2*x + 3
~~~

Try a physics question:

~~~text
A uniform rod of mass M and length L rotates about one end with angular velocity omega. Determine its angular momentum.
~~~

The solution should reach:

~~~text
L = (1/3) M L^2 omega
~~~

Then use the follow-up field and ask another question. Previous turns should remain available to the same question session.

## 12. Third-party software

PrismOffline does not redistribute the Wolfram Engine, llama.cpp binaries, or Qwen GGUF model. Download them from the official sources linked in the setup section.

Their respective licenses and activation terms apply independently.

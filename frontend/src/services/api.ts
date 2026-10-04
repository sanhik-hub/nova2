export type ReasonResponse = {
  success: boolean;
  answer?: unknown;
  error?: string | null;
  engine?: string | null;
   tool_calls?: unknown[];
  cached?: boolean;
  result_only?: boolean;
  elapsed_ms?: number;
  final_solution_elapsed_ms?: number;
};

export type ComputationResponse = {
  success: boolean;
  operation: string;
  input: string;
  result?: string | null;
  error?: string | null;
  engine?: string | null;
  steps?: unknown[];
  plots?: unknown[];
  related?: unknown[];
};

async function parseResponse<T>(response: Response): Promise<T> {
  const raw = await response.text();
  let data: unknown;

  try {
    data = JSON.parse(raw);
  } catch {
    const body = raw.trim().replace(/\\s+/g, " ");
    const detail = body.length > 240 ? body.slice(0, 240) + "…" : body;
    throw new Error(
      detail
        ? `The server returned a non-JSON response (${response.status}): ${detail}`
        : `The server returned an empty non-JSON response (${response.status}).`,
    );
  }

  if (!response.ok) {
    if (
      typeof data === "object" &&
      data !== null &&
      "detail" in data &&
      typeof data.detail === "string"
    ) {
      throw new Error(data.detail);
    }

    if (
      typeof data === "object" &&
      data !== null &&
      "error" in data &&
      typeof data.error === "string"
    ) {
      throw new Error(data.error);
    }

    throw new Error("The server could not complete the request.");
  }

  return data as T;
}

export type ExplanationPreference = "off" | "concise" | "detailed";

export async function reasonQuestion(
  query: string,
  explanation: ExplanationPreference,
): Promise<ReasonResponse> {
  const response = await fetch("/api/reason", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      explanation,
    }),
  });

  return parseResponse<ReasonResponse>(response);
}

export async function computationRequest(
  url: string,
  body: Record<string, unknown>,
): Promise<ComputationResponse> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  return parseResponse<ComputationResponse>(response);
}


export type GuidedOption = { text: string };

export type GuidedStep = {
  question: string;
  options: GuidedOption[];
  correct_index: number;
  hint?: string | null;
};

export type GuidedAttemptResponse = {
  success: boolean;
  query: string;
  steps?: GuidedStep[];
  include_hints?: boolean;
  include_full_solution?: boolean;
  elapsed_ms?: number;
  mcq_elapsed_ms?: number;
  hint_elapsed_ms?: number;
  solution_generation_elapsed_ms?: number;
  full_solution?: string | null;
  error?: string | null;
};

export async function startGuidedAttempt(
  query: string,
  includeHints = false,
  includeFullSolution = true,
): Promise<GuidedAttemptResponse> {
  const response = await fetch("/api/guided/start", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      include_hints: includeHints,
      include_full_solution: includeFullSolution,
    }),
  });

  return parseResponse<GuidedAttemptResponse>(response);
}


export async function generateGuidedSolution(
  query: string,
  steps: GuidedStep[],
): Promise<ReasonResponse> {
  const response = await fetch("/api/guided/solution", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      steps,
    }),
  });

  return parseResponse<ReasonResponse>(response);
}


export type PlotPoint = [number, number];

export type PlotResponse = {
  success: boolean;
  operation: "plot" | string;
  input: string;
  result?: string | null;
  error?: string | null;
  engine?: string | null;
  plots?: Array<{
    type?: string;
    variable?: string;
    points?: PlotPoint[];
  }>;
};

export async function plotExpression(
  expression: string,
  variable = "x",
  xmin = -10,
  xmax = 10,
): Promise<PlotResponse> {
  const response = await fetch("/api/plot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      expression,
      variable,
      xmin,
      xmax,
    }),
  });

  return parseResponse<PlotResponse>(response);
}

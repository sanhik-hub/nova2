export type ReasonResponse = {
  success: boolean;
  answer?: unknown;
  error?: string | null;
  engine?: string | null;
   tool_calls?: unknown[];
  cached?: boolean;
  elapsed_ms?: number;
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
  let data: unknown;

  try {
    data = await response.json();
  } catch {
    throw new Error("The server returned an invalid response.");
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
  error?: string | null;
};

export async function startGuidedAttempt(
  query: string,
  includeHints = false,
  includeFullSolution = false,
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

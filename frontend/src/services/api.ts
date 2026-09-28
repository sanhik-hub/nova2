export type ReasonResponse = {
  success: boolean;
  answer?: unknown;
  error?: string | null;
  engine?: string | null;
  steps?: unknown[];
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

export async function reasonQuestion(
  query: string,
  explanation: boolean,
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

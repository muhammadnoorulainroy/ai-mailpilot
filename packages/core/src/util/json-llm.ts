/**
 * Helpers for parsing JSON the LLM produces inside or outside code fences.
 */

/**
 * Strips a leading and trailing markdown code fence from text, returning the inner content.
 */
export function stripCodeFence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed.startsWith('```')) return trimmed;
  return trimmed
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
}

/**
 * Removes a reasoning model's leading think wrapper, including empty and unclosed forms. Anchored to
 * the start of the response so a literal wrapper inside a JSON string value is never touched.
 */
export function stripThink(text: string): string {
  return text.replace(/^\s*<think>[\s\S]*?<\/think>\s*/i, '').replace(/^\s*<think>[\s\S]*$/i, '');
}

/**
 * Error thrown when an LLM response cannot be parsed as JSON. Carries the raw response.
 */
export class LlmJsonParseError extends Error {
  /**
   * Builds the error, keeping the raw response and optional underlying cause for debugging.
   */
  constructor(
    message: string,
    public readonly raw: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = 'LlmJsonParseError';
  }
}

/**
 * Parses JSON from an LLM response, stripping any code fence first. Throws LlmJsonParseError on failure.
 */
export function parseLlmJson(raw: string): unknown {
  const cleaned = stripCodeFence(raw);
  try {
    return JSON.parse(cleaned);
  } catch (err) {
    throw new LlmJsonParseError(
      `LLM response was not valid JSON: ${err instanceof Error ? err.message : String(err)}`,
      raw,
      err,
    );
  }
}

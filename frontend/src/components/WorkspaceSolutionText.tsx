import type { ReactNode } from "react";
import MathText, {
  normalizeMathText,
} from "./MathText";

type WorkspaceSolutionTextProps = {
  children: string;
  className?: string;
};

function renderLine(line: string, lineIndex: number): ReactNode {
  return (
    <MathText
      key={`math-line-${lineIndex}`}
    >
      {normalizeMathText(line)}
    </MathText>
  );
}

function splitSolutionIntoParagraphs(solution: string): string[] {
  const normalized = String(solution)
    .replace(/\r\n?/g, "\n")
    .trim();

  if (!normalized) {
    return [];
  }

  const steps = normalized
    .split(/(?=\bStep\s+\d+\s*:)/gi)
    .map((part) => part.trim())
    .filter(Boolean);

  if (steps.length > 1) {
    return steps;
  }

  return normalized
    .split(/\n\s*\n+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

export default function WorkspaceSolutionText({
  children,
  className,
}: WorkspaceSolutionTextProps) {
  const paragraphs = splitSolutionIntoParagraphs(children);

  return (
    <div
      className={
        className
          ? `workspace-solution-text ${className}`
          : "workspace-solution-text"
      }
    >
      {paragraphs.map((paragraph, index) => (
        <div
          className="workspace-solution-line"
          key={index}
        >
          {renderLine(paragraph, index)}
        </div>
      ))}
    </div>
  );
}

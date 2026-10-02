import { Component, type ReactNode } from "react";
import katex from "katex";

import "katex/dist/katex.min.css";

type MathTextProps = {
  children: string;
  className?: string;
};

type TextSegment = {
  value: string;
  math: boolean;
  display?: boolean;
};

// -----------------------------------------------------------------------------
// Chemical formula detection
// -----------------------------------------------------------------------------

const CHEM_TOKEN =
  /\b(\d*)([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?){1,})\b/g;

const ELEMENT_SYMBOLS = new Set([
  "H","He","Li","Be","B","C","N","O","F","Ne",
  "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
  "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
  "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
  "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
  "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
  "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
  "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
  "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
  "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
  "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
  "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]);

function startsWithElement(value: string): boolean {
  return (
    ELEMENT_SYMBOLS.has(value.slice(0, 2)) ||
    ELEMENT_SYMBOLS.has(value.slice(0, 1))
  );
}

function renderChemSubscripts(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(CHEM_TOKEN)) {
    const start = match.index ?? 0;
    const full = match[0];
    const coeff = match[1];
    const formula = match[2];

    if (!startsWithElement(formula)) {
      continue;
    }

    if (start > cursor) {
      nodes.push(text.slice(cursor, start));
    }

    if (coeff) {
      nodes.push(coeff);
    }

    const formulaParts: ReactNode[] = [];
    let index = 0;

    while (index < formula.length) {
      const elementMatch = /^([A-Z][a-z]?)/.exec(
        formula.slice(index),
      );

      if (!elementMatch) {
        formulaParts.push(formula[index]);
        index++;
        continue;
      }

      const element = elementMatch[1];
      index += element.length;
      formulaParts.push(element);

      const numberMatch = /^(\d+)/.exec(
        formula.slice(index),
      );

      if (numberMatch) {
        formulaParts.push(
          <sub key={`${start}-${index}`}>
            {numberMatch[1]}
          </sub>,
        );
        index += numberMatch[1].length;
      }
    }

    nodes.push(
      <span key={`chem-${start}`}>
        {formulaParts}
      </span>,
    );

    cursor = start + full.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? nodes : [text];
}

// -----------------------------------------------------------------------------
// Normalization
// -----------------------------------------------------------------------------

export function normalizeMathText(value: string): string {
  return value
    .replace(/−/g, "-")
    .replace(/∗/g, "*")
    .replace(/×/g, "*")
    .replace(/\b1sqrt\b/gi, "1/sqrt")
    .replace(/\bddx\b/gi, "d/dx")
    .replace(/\bdudx\b/gi, "du/dx")
    .replace(/\bdydx\b/gi, "dy/dx")
    .replace(/\bdydu\b/gi, "dy/du")
    .replace(/\bxa\b/g, "x/a")
    .replace(/\b1a\b/g, "1/a")
    .replace(/\ba∗sqrt\b/gi, "a*sqrt")
    .replace(/\ba\*sqrt\b/gi, "a*sqrt");
}

// -----------------------------------------------------------------------------
// Plain mathematical-expression conversion
// -----------------------------------------------------------------------------

function convertToLatex(value: string): string {
  let latex = normalizeMathText(value.trim())
    .replace(/>\s*=/g, "\\geq")
    .replace(/<=/g, "\\leq")
    .replace(/>=/g, "\\geq")
    .replace(/!=/g, "\\neq")
    .replace(/\+\/−|\+\/-/g, "\\pm")
    .replace(/±/g, "\\pm")
    .replace(/\*/g, "\\cdot")
    .replace(/÷/g, "\\div")
    .replace(/\bpi\b/gi, "\\pi");

  latex = latex.replace(
    /sqrt\(([^()]+)\)/gi,
    "\\sqrt{$1}",
  );

  latex = latex.replace(
    /([A-Za-z0-9])\^\s*(?:\{([^}]+)\}|([A-Za-z0-9]+))/g,
    (_match, base: string, braced?: string, plain?: string) =>
      `${base}^{${braced ?? plain}}`,
  );

  latex = latex.replace(
    /([A-Za-z0-9])_\s*(?:\{([^}]+)\}|([A-Za-z0-9]+))/g,
    (_match, base: string, braced?: string, plain?: string) =>
      `${base}_{${braced ?? plain}}`,
  );

  latex = latex.replace(
    /\b([A-Za-z0-9]+)\s*\/\s*([A-Za-z0-9]+)\b/g,
    "\\frac{$1}{$2}",
  );

  return latex;
}

// -----------------------------------------------------------------------------
// KaTeX renderer
// -----------------------------------------------------------------------------

class MathErrorBoundary extends Component<
  {
    original: string;
    children: ReactNode;
  },
  {
    failed: boolean;
  }
> {
  state = {
    failed: false,
  };

  static getDerivedStateFromError(): { failed: boolean } {
    return {
      failed: true,
    };
  }

  render() {
    if (this.state.failed) {
      return <>{this.props.original}</>;
    }

    return this.props.children;
  }
}

function RenderedMath({
  value,
  display = false,
  convert = false,
}: {
  value: string;
  display?: boolean;
  convert?: boolean;
}) {
  const source = convert
    ? convertToLatex(value)
    : value.trim().replace(/`/g, "");

  return (
    <MathErrorBoundary original={value}>
      <span
        className={display ? "math-text-display" : "math-text-inline"}
        dangerouslySetInnerHTML={{
          __html: katex.renderToString(source, {
            displayMode: display,
            throwOnError: false,
            strict: false,
            trust: false,
          }),
        }}
      />
    </MathErrorBoundary>
  );
}

// -----------------------------------------------------------------------------
// Explicit Markdown/LaTeX delimiters
// -----------------------------------------------------------------------------

function splitDelimitedMath(value: string): TextSegment[] {
  const segments: TextSegment[] = [];
  let cursor = 0;

  const pattern =
    /\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g;

  for (const match of value.matchAll(pattern)) {
    const start = match.index ?? 0;

    if (start > cursor) {
      segments.push({
        value: value.slice(cursor, start),
        math: false,
      });
    }

    const display = match[1] !== undefined;
    const mathValue = display
      ? match[1]
      : match[2];

    segments.push({
      value: mathValue,
      math: true,
      display,
    });

    cursor = start + match[0].length;
  }

  if (cursor < value.length) {
    segments.push({
      value: value.slice(cursor),
      math: false,
    });
  }

  return segments;
}

// -----------------------------------------------------------------------------
// Legacy plain-text math detection from Prism
// -----------------------------------------------------------------------------

const MATH_FRAGMENT =
  /(?:sqrt\([^()]+\)|[A-Za-z0-9]+(?:\s*(?:\^|_)\s*(?:\{[^}]+\}|[A-Za-z0-9]+))|(?:[A-Za-z0-9]+(?:\s*(?:\/|<=|>=|>\s*=|!=|=|\+|-|\*|×|÷)\s*[A-Za-z0-9]+)+)|\+\/−|\+\/-|<=|>=|>\s*=|!=|\bpi\b)/gi;

function splitLegacyMath(value: string): TextSegment[] {
  const segments: TextSegment[] = [];
  let cursor = 0;

  for (const match of value.matchAll(MATH_FRAGMENT)) {
    const start = match.index ?? 0;
    const fragment = match[0];

    if (start > cursor) {
      segments.push({
        value: value.slice(cursor, start),
        math: false,
      });
    }

    segments.push({
      value: fragment,
      math: true,
      display: false,
    });

    cursor = start + fragment.length;
  }

  if (cursor < value.length) {
    segments.push({
      value: value.slice(cursor),
      math: false,
    });
  }

  return segments.length > 0
    ? segments
    : [{ value, math: false }];
}

// -----------------------------------------------------------------------------
// Public renderer
// -----------------------------------------------------------------------------

export default function MathText({
  children,
  className,
}: MathTextProps) {
  const normalized = normalizeMathText(String(children));

  // First handle explicit $...$ / $$...$$ generated by the local model.
  const delimited = splitDelimitedMath(normalized);

  // If explicit delimiters were found, preserve them exactly.
  const hasDelimitedMath = delimited.some(
    (segment) => segment.math,
  );

  const segments = hasDelimitedMath
    ? delimited
    : splitLegacyMath(normalized);

  return (
    <span className={className ?? "math-text"}>
      {segments.map((segment, index) => {
        if (segment.math) {
          return (
            <RenderedMath
              key={`${index}-${segment.value}`}
              value={segment.value}
              display={segment.display}
              convert={!hasDelimitedMath}
            />
          );
        }

        return (
          <span key={`${index}-${segment.value}`}>
            {renderChemSubscripts(segment.value).map(
              (node, nodeIndex) => (
                <span key={nodeIndex}>{node}</span>
              ),
            )}
          </span>
        );
      })}
    </span>
  );
}

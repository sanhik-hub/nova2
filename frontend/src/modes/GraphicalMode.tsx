import type { PointerEvent as ReactPointerEvent } from "react";
import { useMemo, useState } from "react";
import { plotExpression, type PlotPoint } from "../services/api";
import { LoadingState } from "../components/LoadingState";
import WorkspaceSolutionText from "../components/WorkspaceSolutionText";

type Coordinate = [number, number];

function GraphCanvas({
  coordinates,
  curve,
  solutionPoints,
  onChange,
}: {
  coordinates: Coordinate[];
  curve: PlotPoint[];
  solutionPoints: Coordinate[];
  onChange: (points: Coordinate[]) => void;
}) {
  const size = 540;
  const padding = 30;
  const origin = size / 2;
  const plotSize = size - padding * 2;
  const axisLimit = 20;
  const unit = plotSize / (axisLimit * 2);

  const toSvg = ([x, y]: Coordinate): [number, number] => [
    origin + x * unit,
    origin - y * unit,
  ];

  function movePoint(index: number, event: ReactPointerEvent<SVGCircleElement>) {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;

    const update = (moveEvent: PointerEvent) => {
      const rect = svg.getBoundingClientRect();
      const sx = ((moveEvent.clientX - rect.left) / rect.width) * size;
      const sy = ((moveEvent.clientY - rect.top) / rect.height) * size;

      const next: Coordinate = [
        Math.max(-axisLimit, Math.min(axisLimit, Math.round(((sx - origin) / unit) * 2) / 2)),
        Math.max(-axisLimit, Math.min(axisLimit, Math.round(((origin - sy) / unit) * 2) / 2)),
      ];

      onChange(coordinates.map((point, i) => (i === index ? next : point)));
    };

    const stop = () => {
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", stop);
    };

    window.addEventListener("pointermove", update);
    window.addEventListener("pointerup", stop, { once: true });
  }

  const solutionPointSvg = solutionPoints.map(toSvg);

  const curvePoints = curve
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    .map(toSvg)
    .map(([x, y]) => `${x},${y}`)
    .join(" ");

  return (
    <svg className="graphical-canvas" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Coordinate graph">
      <rect width={size} height={size} className="graphical-background" />

      {Array.from({ length: 21 }, (_, index) => {
        const value = -axisLimit + index * 2;
        const position = padding + index * (plotSize / 20);

        return (
          <g key={index}>
            <line x1={position} y1={padding} x2={position} y2={size - padding} className="graphical-grid-line" />
            <line x1={padding} y1={position} x2={size - padding} y2={position} className="graphical-grid-line" />
            {value !== 0 && value % 4 === 0 && (
              <>
                <text x={position} y={origin + 16} textAnchor="middle" className="graphical-coordinate-label">{value}</text>
                <text x={origin - 7} y={origin - value * unit + 4} textAnchor="end" className="graphical-coordinate-label">{value}</text>
              </>
            )}
          </g>
        );
      })}

      <line x1={padding} y1={origin} x2={size - padding} y2={origin} className="graphical-axis" />
      <line x1={origin} y1={size - padding} x2={origin} y2={padding} className="graphical-axis" />
      <text x={size - padding + 4} y={origin - 7} className="graphical-axis-label">x</text>
      <text x={origin + 8} y={padding - 7} className="graphical-axis-label">y</text>
      <text x={origin + 8} y={origin + 16} className="graphical-axis-label">O</text>

      {curvePoints && <polyline points={curvePoints} className="graphical-curve" />}

      {solutionPointSvg.map(([x, y], index) => (
        <g key={`solution-${index}`}>
          <circle cx={x} cy={y} r="7" className="graphical-solution-point" />
          <text x={x + 10} y={y + 18} className="graphical-point-label">S{index + 1}</text>
        </g>
      ))}

      {coordinates.map((point, index) => {
        const [x, y] = toSvg(point);
        return (
          <g key={index}>
            <circle
              cx={x}
              cy={y}
              r="8"
              className="graphical-point"
              onPointerDown={(event) => movePoint(index, event)}
            />
            <text x={x + 10} y={y - 10} className="graphical-point-label">
              P{index + 1} ({point[0]}, {point[1]})
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function GraphicalMode() {
  const [expression, setExpression] = useState("");
  const [variable, setVariable] = useState("x");
  const [xmin, setXmin] = useState("-10");
  const [xmax, setXmax] = useState("10");
  const [coordinates, setCoordinates] = useState<Coordinate[]>([]);
  const [curve, setCurve] = useState<PlotPoint[]>([]);
  const [solutionPoints, setSolutionPoints] = useState<Coordinate[]>([]);
  const [solutionText, setSolutionText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canPlot = expression.trim().length > 0 && !loading;

  async function plot() {
    if (!canPlot) return;

    setLoading(true);
    setError(null);

    try {
      const result = await plotExpression(
        expression.trim(),
        variable.trim() || "x",
        Number(xmin),
        Number(xmax),
      );

      if (!result.success) {
        setCurve([]);
        setSolutionPoints([]);
        setSolutionText("");
        setError(result.error || "Unable to generate the graph.");
        return;
      }

      const points = result.plots?.[0]?.points;
      if (!Array.isArray(points)) {
        setCurve([]);
        setSolutionPoints([]);
        setSolutionText("");
        setError("The graph engine returned no plot data.");
        return;
      }

      setCurve(points);
      setSolutionPoints((result.solution_points ?? []).map((point) => [Number(point[0]), Number(point[1])] as Coordinate));
      setSolutionText(result.result ?? "");
    } catch (err) {
      setCurve([]);
      setError(err instanceof Error ? err.message : "Unable to generate the graph.");
    } finally {
      setLoading(false);
    }
  }

  const pointCountLabel = useMemo(
    () => `${coordinates.length}/4 points`,
    [coordinates.length],
  );

  return (
    <section className="workspace">
      <div className="page-heading">
        <div>
          <div className="eyebrow">GRAPHICAL MODE</div>
          <h1>Explore a graph</h1>
          <p>Enter an equation or expression, plot it, and inspect its solution points.</p>
        </div>
      </div>

      <div className="graphical-grid">
        <div className="card graphical-controls">
          <label className="field-label" htmlFor="graph-expression">Equation or expression</label>
          <input
            id="graph-expression"
            className="question-input graphical-expression-input"
            value={expression}
            onChange={(event) => setExpression(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void plot();
            }}
            placeholder="For example: x^2 - 4"
          />

          <div className="graphical-fields">
            <div>
              <label className="field-label" htmlFor="graph-variable">Variable</label>
              <input id="graph-variable" className="text-input" value={variable} onChange={(event) => setVariable(event.target.value)} />
            </div>
            <div>
              <label className="field-label" htmlFor="graph-xmin">Minimum x</label>
              <input id="graph-xmin" className="text-input" type="number" value={xmin} onChange={(event) => setXmin(event.target.value)} />
            </div>
            <div>
              <label className="field-label" htmlFor="graph-xmax">Maximum x</label>
              <input id="graph-xmax" className="text-input" type="number" value={xmax} onChange={(event) => setXmax(event.target.value)} />
            </div>
          </div>

          <div className="graph-editor-header">
            <div>
              <strong>Coordinate points</strong>
              <span>{pointCountLabel}</span>
            </div>
            <div className="graph-editor-actions">
              <button className="graph-control" type="button" onClick={() => setCoordinates([...coordinates, [0, 0]])} disabled={coordinates.length >= 4}>+ Add point</button>
              <button className="graph-control" type="button" onClick={() => setCoordinates(coordinates.slice(0, -1))} disabled={coordinates.length === 0}>− Remove point</button>
            </div>
          </div>

          <p className="graph-help">Drag points on the 20 × 20 grid. Points snap to 0.5-unit increments.</p>

          <button className="primary-button" type="button" onClick={() => void plot()} disabled={!canPlot}>
            {loading ? "Plotting..." : "Solve & Plot"}
          </button>

          {error && <p className="error-message">{error}</p>}
        </div>

        <div className="card graphical-plot-card">
          <GraphCanvas coordinates={coordinates} curve={curve} solutionPoints={solutionPoints} onChange={setCoordinates} />
          <p className="graph-help">The curve is sampled by the local exact computation engine.</p>
          {solutionText && <div className="graphical-solution-card"><div className="answer-label">Solution</div><div className="answer-content"><WorkspaceSolutionText>{solutionText}</WorkspaceSolutionText></div></div>}
        </div>
      </div>

      {loading && <LoadingState text="Generating the graph..." />}
    </section>
  );
}
